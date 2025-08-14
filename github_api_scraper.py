#!/usr/bin/env python3
"""
GitHub API Scraper for AI Code Generator Markers
"""
import os
import sys
import json
import time
import requests
import math
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
from pathlib import Path
from collections import Counter
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, func, UniqueConstraint, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
import psycopg
from github import Github
import re
import ast

# Configure logging for the module
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database setup - PostgreSQL
# Use environment variable for DATABASE_URL, fallback to SQLite for local development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ai_code_generator.db")

# Configure engine based on database type
if DATABASE_URL.startswith("postgresql"):
    # For now, use direct connection instead of pooling to test connectivity
    engine = create_engine(
        DATABASE_URL.replace("postgresql://", "postgresql+psycopg://"), 
        pool_pre_ping=True, 
        pool_recycle=300,
        pool_size=3,  # Reduced from 5 to 3
        max_overflow=5,  # Reduced from 10 to 5
        pool_timeout=30,  # Add timeout
        echo=False  # Disable SQL echo for production
    )
else:
    # SQLite configuration (for local development)
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class MarkerHit(Base):
    __tablename__ = "marker_hits"
    id = Column(Integer, primary_key=True, index=True)
    marker = Column(String, index=True)
    repo_name = Column(String, index=True)
    repo_url = Column(String)
    file_path = Column(String)
    file_url = Column(String)
    stars = Column(Integer)
    description = Column(String)
    owner_type = Column(String, index=True)
    owner_login = Column(String, index=True)
    # Contact information fields
    owner_email = Column(String)
    contact_source = Column(String)  # 'github_profile', 'repo_content', or 'none'
    contact_extracted_at = Column(String, default=lambda: datetime.utcnow().isoformat())
    # Repository activity fields
    latest_commit_date = Column(String)
    # Top contributor fields
    top_contributor = Column(String, index=True)
    top_contributor_email = Column(String)
    # Pagination tracking fields
    scraping_page = Column(Integer)
    scraping_position = Column(Integer)
    last_scraped_at = Column(DateTime, default=datetime.utcnow)
    
    # Add unique constraint to prevent duplicates
    __table_args__ = (
        UniqueConstraint('marker', 'repo_name', 'file_path', name='unique_marker_repo_file'),
    )

# Database initialization function
def init_database():
    """Initialize database tables. Only call this when needed, not during import."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        # Don't raise the exception - let the application continue
        pass

def migrate_database():
    """Run database migration to add new columns. Safe to run multiple times."""
    try:
        logger.info("Starting database migration...")
        
        with SessionLocal() as session:
            logger.info("Database session created successfully")
            
            # Use a separate connection to check if columns exist without affecting the main transaction
            engine = session.get_bind()
            with engine.connect() as connection:
                # Check if columns already exist using information_schema
                logger.info("Checking if columns exist using information_schema...")
                result = connection.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'marker_hits' 
                    AND column_name IN ('top_contributor', 'top_contributor_email', 'scraping_page', 'scraping_position', 'last_scraped_at')
                """)).fetchall()
                
                existing_columns = [row[0] for row in result]
                logger.info(f"Existing columns found: {existing_columns}")
                
                # Check for top contributor columns
                if 'top_contributor' in existing_columns and 'top_contributor_email' in existing_columns:
                    logger.info("Top contributor columns already exist")
                else:
                    # Add the top contributor columns using raw SQL
                    if 'top_contributor' not in existing_columns:
                        logger.info("Adding top_contributor column...")
                        session.execute(text("ALTER TABLE marker_hits ADD COLUMN IF NOT EXISTS top_contributor VARCHAR"))
                        logger.info("top_contributor column added")
                    
                    if 'top_contributor_email' not in existing_columns:
                        logger.info("Adding top_contributor_email column...")
                        session.execute(text("ALTER TABLE marker_hits ADD COLUMN IF NOT EXISTS top_contributor_email VARCHAR"))
                        logger.info("top_contributor_email column added")
                
                # Check for pagination columns
                if 'scraping_page' in existing_columns and 'scraping_position' in existing_columns and 'last_scraped_at' in existing_columns:
                    logger.info("Pagination columns already exist")
                else:
                    # Add the pagination columns using raw SQL
                    if 'scraping_page' not in existing_columns:
                        logger.info("Adding scraping_page column...")
                        session.execute(text("ALTER TABLE marker_hits ADD COLUMN IF NOT EXISTS scraping_page INTEGER"))
                        logger.info("scraping_page column added")
                    
                    if 'scraping_position' not in existing_columns:
                        logger.info("Adding scraping_position column...")
                        session.execute(text("ALTER TABLE marker_hits ADD COLUMN IF NOT EXISTS scraping_position INTEGER"))
                        logger.info("scraping_position column added")
                    
                    if 'last_scraped_at' not in existing_columns:
                        logger.info("Adding last_scraped_at column...")
                        session.execute(text("ALTER TABLE marker_hits ADD COLUMN IF NOT EXISTS last_scraped_at TIMESTAMP"))
                        logger.info("last_scraped_at column added")
            
            logger.info("Committing changes...")
            session.commit()
            logger.info("Changes committed successfully")
            
            # Verify the new columns exist
            logger.info("Verifying new columns...")
            result = session.execute(text("SELECT top_contributor, top_contributor_email, scraping_page, scraping_position, last_scraped_at FROM marker_hits LIMIT 1")).fetchall()
            logger.info(f"New columns verified successfully. Result: {result}")
            return True
            
    except Exception as e:
        logger.error(f"Error during database migration: {e}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {str(e)}")
        return False

class ScrapingStateManager:
    """Manages pagination state for scraping operations."""
    
    def __init__(self, db_session):
        self.db = db_session
    
    def get_scraping_state(self, marker: str) -> dict:
        """Get current scraping state for a marker"""
        try:
            # Query the most recent record for this marker to get pagination state
            latest_record = self.db.query(MarkerHit).filter(
                MarkerHit.marker == marker,
                MarkerHit.scraping_page.isnot(None)
            ).order_by(MarkerHit.last_scraped_at.desc()).first()
            
            if latest_record:
                return {
                    'page': latest_record.scraping_page,
                    'position': latest_record.scraping_position,
                    'last_repo': latest_record.repo_name,
                    'last_updated': latest_record.last_scraped_at
                }
            
            # No pagination state exists - check if we have any data for this marker
            existing_count = self.db.query(MarkerHit).filter(
                MarkerHit.marker == marker
            ).count()
            
            if existing_count > 0:
                logger.info(f"Found {existing_count} existing records for {marker}, starting fresh pagination")
            
            return {'page': 1, 'position': 0, 'last_repo': None, 'last_updated': None}
            
        except Exception as e:
            logger.error(f"Error getting scraping state for {marker}: {e}")
            return {'page': 1, 'position': 0, 'last_repo': None, 'last_updated': None}
    
    def update_scraping_state(self, marker: str, page: int, position: int, repo_name: str):
        """Update scraping state by creating a state tracking record"""
        try:
            # Create a state tracking record
            state_record = MarkerHit(
                marker=marker,
                repo_name=f"STATE_TRACKING_{marker}",  # Special identifier
                repo_url="",
                file_path="",
                file_url="",
                stars=0,
                description="",
                owner_type="",
                owner_login="",
                scraping_page=page,
                scraping_position=position,
                last_scraped_at=datetime.utcnow()
            )
            
            try:
                self.db.add(state_record)
                self.db.commit()
                logger.debug(f"Updated scraping state for {marker}: page {page}, position {position}")
            except IntegrityError:
                # If duplicate, update existing state record
                existing = self.db.query(MarkerHit).filter(
                    MarkerHit.marker == marker,
                    MarkerHit.repo_name == f"STATE_TRACKING_{marker}"
                ).first()
                if existing:
                    existing.scraping_page = page
                    existing.scraping_position = position
                    existing.last_scraped_at = datetime.utcnow()
                    self.db.commit()
                    logger.debug(f"Updated existing scraping state for {marker}: page {page}, position {position}")
                    
        except Exception as e:
            logger.error(f"Error updating scraping state for {marker}: {e}")
            self.db.rollback()
    
    def reset_scraping_state(self, marker: str):
        """Reset scraping state for a marker (start fresh)"""
        try:
            # Delete existing state tracking record
            self.db.query(MarkerHit).filter(
                MarkerHit.marker == marker,
                MarkerHit.repo_name == f"STATE_TRACKING_{marker}"
            ).delete()
            self.db.commit()
            logger.info(f"Reset scraping state for {marker}")
        except Exception as e:
            logger.error(f"Error resetting scraping state for {marker}: {e}")
            self.db.rollback()



@dataclass
class FileAnalysis:
    """Data class for storing file analysis results."""
    repo_name: str
    file_path: str
    language: str
    lines_of_code: int
    file_size: int
    comment_density: float
    naming_entropy: float
    pattern_repetition: float
    complexity_score: float
    suspicious_patterns: List[str]
    overall_score: float

@dataclass
class RepoAnalysis:
    """Data class for storing repository analysis results."""
    repo_name: str
    total_files_analyzed: int
    total_files_in_repo: int
    languages: Dict[str, int]
    avg_comment_density: float
    avg_naming_entropy: float
    avg_pattern_repetition: float
    avg_complexity: float
    suspicious_files: List[str]
    overall_score: float
    analysis_coverage: float  # Percentage of files analyzed

class GitHubAPIScraper:
    """
    Handles GitHub API interactions without cloning repositories.
    Provides methods to search for repositories, fetch file contents, and handle rate limits.
    """
    
    def __init__(self, token: Optional[str] = None, backup_tokens: Optional[List[str]] = None):
        # Primary token (most common use case)
        self.primary_token = token
        
        # Backup tokens (optional, for advanced users)
        self.backup_tokens = backup_tokens or []
        self.all_tokens = [token] if token else []
        if backup_tokens:
            self.all_tokens.extend(backup_tokens)
        
        self.current_token_index = 0
        self.rate_limit_errors = 0  # Simple counter for rate limit errors
        
        # Initialize with primary token
        if self.primary_token:
            self._initialize_with_token(self.primary_token)
            logger.info(f"GitHub API initialized with token (first 8 chars: {self.primary_token[:8]}...)")
            if self.backup_tokens:
                logger.info(f"Backup tokens available: {len(self.backup_tokens)}")
            
            # Add initial delay to avoid rate limits on startup
            logger.info("Adding initial delay to avoid rate limits...")
            time.sleep(5.0)
        else:
            self.github = Github()
            self.session = requests.Session()
            logger.warning("GitHub API initialized without token - limited rate limits")
        
        # Rate limiting
        self.requests_made = 0
        self.rate_limit_delay = 5.0  # Increased to 5 seconds between requests
        self.secondary_rate_limit_delay = 30.0  # Increased to 30 seconds
        self.search_api_delay = 10.0  # Increased to 10 seconds for search API calls
    
    def _initialize_with_token(self, token: str):
        """Initialize GitHub client and session with a specific token."""
        self.github = Github(token)
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'token {token}'})
        self.current_token = token
    
    def _rotate_token(self):
        """Rotate to the next available token (only if backup tokens exist)."""
        if not self.backup_tokens:
            logger.warning("No backup tokens available for rotation")
            return False
        
        self.current_token_index = (self.current_token_index + 1) % len(self.all_tokens)
        new_token = self.all_tokens[self.current_token_index]
        
        logger.info(f"Rotating to backup token {self.current_token_index + 1}/{len(self.all_tokens)} (first 8 chars: {new_token[:8]}...)")
        self._initialize_with_token(new_token)
        return True
    
    def _handle_rate_limit_error(self, error_msg: str, operation: str = "API call"):
        """
        Handle rate limit errors intelligently.
        Returns True if we should retry the operation, False otherwise.
        """
        # Track rate limit errors
        self.rate_limit_errors += 1
        
        # Check if this is a secondary rate limit (abuse rate limit)
        is_secondary_rate_limit = any(keyword in error_msg.lower() for keyword in [
            'abuse', 'secondary', 'burst', 'too many requests', 'rate limit exceeded'
        ])
        
        if is_secondary_rate_limit:
            logger.warning(f"Secondary rate limit detected for {operation}. Error: {error_msg}")
            
            # For secondary rate limits, try token rotation first (if backup tokens exist)
            if self.backup_tokens:
                logger.info("Attempting token rotation for secondary rate limit...")
                if self._rotate_token():
                    logger.info("Token rotated successfully, will retry operation")
                    return True
            
            # If no backup tokens or rotation failed, wait longer with exponential backoff
            wait_time = self.secondary_rate_limit_delay * (2 ** min(self.rate_limit_errors, 3))
            logger.warning(f"Waiting {wait_time} seconds for secondary rate limit...")
            time.sleep(wait_time)
            return True
        
        else:
            # Primary rate limit - check if we can rotate tokens
            if self.backup_tokens:
                logger.info("Primary rate limit reached, attempting token rotation...")
                if self._rotate_token():
                    logger.info("Token rotated successfully, will retry operation")
                    return True
            
            # If no backup tokens, wait for rate limit reset
            logger.warning(f"Primary rate limit reached for {operation}. Waiting for reset...")
            return False
    
    def check_rate_limit(self):
        """Check and handle GitHub API rate limits. Sleeps if close to limit."""
        try:
            rate_limit = self.github.get_rate_limit()
            remaining = rate_limit.core.remaining
            limit = rate_limit.core.limit
            
                    # Log rate limit status to verify token is working
            if self.primary_token:
                logger.info(f"Rate limit: {remaining}/{limit} remaining (authenticated)")
            else:
                logger.info(f"Rate limit: {remaining}/{limit} remaining (unauthenticated)")
            
            if remaining < 100:  # More conservative threshold
                reset_time = rate_limit.core.reset.timestamp()
                current_time = time.time()
                sleep_time = max(reset_time - current_time, 300)  # Increased minimum sleep to 5 minutes
                
                logger.warning(f"Rate limit low ({remaining} remaining). Sleeping for {sleep_time:.0f} seconds...")
                time.sleep(sleep_time)
            elif remaining < 500:  # More conservative threshold
                logger.info(f"Rate limit getting low ({remaining} remaining). Adding delay...")
                time.sleep(5.0)  # Increased delay
            else:
                # Always add a delay to prevent secondary rate limits
                time.sleep(2.0)
        except Exception as e:
            logger.warning(f"Error checking rate limit: {e}")
            # If we can't check rate limit, add a conservative delay
            time.sleep(2)  # Increased delay
    
    def _make_api_call_with_retry(self, api_call_func, *args, **kwargs):
        """
        Make an API call with automatic retry and token rotation on rate limit errors.
        
        Args:
            api_call_func: Function that makes the API call
            *args, **kwargs: Arguments to pass to the API call function
            
        Returns:
            The result of the API call, or None if all retries failed
        """
        max_retries = 3  # Fixed number of retries regardless of token count
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                result = api_call_func(*args, **kwargs)
                if result is not None:
                    return result
                else:
                    logger.warning(f"API call returned None, retrying... (attempt {retry_count + 1})")
                    retry_count += 1
                    time.sleep(5.0)  # Wait 5 seconds before retry
                    continue
            except Exception as e:
                error_msg = str(e)
                
                # Check if this is a rate limit error
                if "rate limit" in error_msg.lower() or "403" in error_msg:
                    logger.warning(f"Rate limit error on attempt {retry_count + 1}: {error_msg}")
                    
                    # Try token rotation first (if backup tokens exist)
                    if self.backup_tokens:
                        logger.info("Attempting token rotation...")
                        if self._rotate_token():
                            logger.info("Token rotated, retrying...")
                            retry_count += 1
                            time.sleep(2.0)  # Short delay after token rotation
                            continue
                    
                    # If no backup tokens or rotation failed, wait with exponential backoff
                    wait_time = 30 * (2 ** retry_count)  # 30s, 60s, 120s
                    logger.warning(f"Waiting {wait_time} seconds before retry {retry_count + 1}...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                else:
                    # Non-rate-limit error, log and retry once
                    logger.error(f"Non-rate-limit error on attempt {retry_count + 1}: {error_msg}")
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        time.sleep(5.0)
                        continue
                    else:
                        logger.error(f"All retry attempts failed for non-rate-limit error")
                        return None
        
        logger.error(f"All {max_retries} retry attempts failed")
        return None
    
    def search_repositories(self, query: str, max_repos: int = 10, min_stars: int = 0) -> List[Dict]:
        """
        Search for repositories using GitHub API.
        Returns a list of repository metadata dicts, each including a repo object for further API calls.
        """
        try:
            # Add star filter to query if specified
            if min_stars > 0:
                query += f" stars:>={min_stars}"
                
            # Use retry mechanism for the search
            repos = self._make_api_call_with_retry(
                self.github.search_repositories, 
                query=query, sort="stars", order="desc"
            )
            
            if repos is None:
                logger.error("Failed to search repositories after all retries")
                return []
            
            results = []
            
            for i, repo in enumerate(repos):
                if i >= max_repos:
                    break
                
                self.check_rate_limit()
                
                try:
                    results.append({
                        'name': repo.full_name,
                        'url': repo.html_url,
                        'clone_url': repo.clone_url,
                        'language': repo.language,
                        'stars': repo.stargazers_count,
                        'size': repo.size,
                        'description': repo.description,
                        'topics': repo.get_topics(),
                        'default_branch': repo.default_branch,
                        'repo_object': repo  # Keep reference for API calls
                    })
                    logger.info(f"Found repo: {repo.full_name} ({repo.stargazers_count} stars)")
                except Exception as e:
                    logger.warning(f"Error processing repo {repo.full_name}: {e}")
                    continue
            
            return results
        except Exception as e:
            logger.error(f"Error searching repositories: {e}")
            return []
    
    def get_repository_contents(self, repo, path: str = "", max_files: int = 100) -> List[Dict]:
        """
        Get repository contents through API without cloning.
        Recursively fetches code files up to max_files.
        Returns a list of file metadata dicts.
        """
        contents = []
        files_processed = 0
        
        try:
            self.check_rate_limit()
            repo_contents = repo.get_contents(path)
            
            # Handle both single files and directories
            if not isinstance(repo_contents, list):
                repo_contents = [repo_contents]
            
            for content in repo_contents:
                if files_processed >= max_files:
                    break
                
                try:
                    if content.type == "dir":
                        # Skip common non-code directories
                        skip_dirs = {'.git', 'node_modules', '__pycache__', 'venv', 'env',
                                   'build', 'dist', 'target', '.pytest_cache', 'vendor',
                                   'third_party', 'deps', 'bower_components'}
                        
                        if content.name not in skip_dirs and not content.name.startswith('.'):
                            # Recursively get directory contents
                            subcontents = self.get_repository_contents(
                                repo, content.path, max_files - files_processed
                            )
                            contents.extend(subcontents)
                            files_processed += len(subcontents)
                    
                    elif content.type == "file":
                        # Check if it's a code file we want to analyze
                        if self.is_code_file(content.name) and content.size < 1024 * 1024:  # Skip files > 1MB
                            contents.append({
                                'path': content.path,
                                'name': content.name,
                                'size': content.size,
                                'download_url': content.download_url,
                                'content_object': content
                            })
                            files_processed += 1
                
                except Exception as e:
                    logger.warning(f"Error processing content {content.path}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error getting repository contents: {e}")
        
        return contents
    
    def get_file_content(self, content_object) -> Optional[str]:
        """
        Get file content from GitHub API.
        Uses download_url for large files, decoded_content for small files.
        Returns the file content as a string, or None on error.
        """
        try:
            self.check_rate_limit()
            
            # Use download_url for larger files, decoded_content for smaller ones
            if content_object.size > 100 * 1024:  # 100KB threshold
                response = self.session.get(content_object.download_url)
                if response.status_code == 200:
                    return response.text
            else:
                # Use the content API (base64 decoded)
                content = content_object.decoded_content
                if isinstance(content, bytes):
                    return content.decode('utf-8', errors='ignore')
                return content
                
        except Exception as e:
            logger.warning(f"Error getting file content for {content_object.path}: {e}")
        
        return None
    
    def is_code_file(self, filename: str) -> bool:
        """
        Check if file is a code file we want to analyze, based on extension.
        """
        code_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.cc', '.cxx',
            '.hpp', '.h', '.c', '.cs', '.go', '.rs', '.php', '.rb', '.swift',
            '.kt', '.scala', '.m', '.mm', '.pl', '.r', '.sql', '.sh', '.bash',
            '.ps1', '.vbs', '.lua', '.dart', '.elm', '.clj', '.hs', '.ml',
            '.fs', '.pas', '.ada', '.cob', '.for', '.f90', '.jl', '.nim'
        }
        
        return re.match(r'\.(py|js|jsx|ts|tsx|java|cpp|cc|cxx|hpp|h|c|cs|go|rs|php|rb|swift|kt|scala|m|mm|pl|r|sql|sh|bash|ps1|vbs|lua|dart|elm|clj|hs|ml|fs|pas|ada|cob|for|f90|jl|nim)$', filename, re.IGNORECASE) is not None

    def _get_first_search_result(self, marker: str, min_stars: int = 0) -> Optional[dict]:
        """Get just the first search result to check if we should resume"""
        try:
            query = f'path:{marker}'
            if min_stars > 0:
                query += f' stars:>={min_stars}'
            
            # Check rate limit and add delays
            self.check_rate_limit()
            time.sleep(self.search_api_delay)
            
            # Use retry mechanism for search API calls
            results = self._make_api_call_with_retry(
                self.github.search_code, 
                query=query, per_page=1
            )
            
            if results and hasattr(results, 'totalCount') and results.totalCount > 0:
                first_file = results[0]
                logger.info(f"First result for {marker}: {first_file.repository.full_name}")
                return {
                    'repo_name': first_file.repository.full_name,
                    'file_path': first_file.path
                }
            elif results and len(results) > 0:
                first_file = results[0]
                logger.info(f"First result for {marker}: {first_file.repository.full_name}")
                return {
                    'repo_name': first_file.repository.full_name,
                    'file_path': first_file.path
                }
            else:
                logger.warning(f"No results found for {marker} (API call may have failed)")
        except Exception as e:
            logger.warning(f"Error getting first result for {marker}: {e}")
        
        return None

    def _is_repo_in_database(self, repo_name: str) -> bool:
        """Check if repository already exists in database"""
        try:
            with SessionLocal() as session:
                existing = session.query(MarkerHit).filter(
                    MarkerHit.repo_name == repo_name
                ).first()
                return existing is not None
        except Exception as e:
            logger.error(f"Error checking if repo {repo_name} exists in database: {e}")
            return False

    def _get_search_results_page(self, query: str, page: int) -> List:
        """Get search results for a specific page"""
        try:
            # Check rate limit and add delays
            self.check_rate_limit()
            time.sleep(self.search_api_delay)
            
            # Use retry mechanism for search API calls
            results = self._make_api_call_with_retry(
                self.github.search_code, 
                query=query, per_page=30, page=page
            )
            
            if results and hasattr(results, 'totalCount'):
                logger.info(f"Search query '{query}' page {page}: {results.totalCount} total results")
                return list(results)
            elif results:
                logger.info(f"Search query '{query}' page {page}: {len(results)} results")
                return list(results)
            else:
                logger.warning(f"Search query '{query}' page {page}: No results returned (API call may have failed)")
                return []
        except Exception as e:
            logger.error(f"Error getting page {page} for query '{query}': {e}")
            return []

    def search_ai_code_generator_files(self, max_repos_per_pattern: int = 10, min_stars: int = 0, existing_data: dict = None) -> dict:
        """
        Search GitHub for repositories containing files that are markers for AI code generators.
        Skips existing results to get new data points.

        Args:
            max_repos_per_pattern: Maximum repositories to return per marker pattern.
            min_stars: Minimum number of stars for repositories to include.
            existing_data: Dictionary of existing results to skip (format: {marker: [{'repo_name': ..., 'file_path': ...}, ...]})

        Usage:
            scraper = GitHubAPIScraper(token)
            results = scraper.search_ai_code_generator_files(existing_data=your_existing_data)
        """
        # List of AI code generator marker files to search for
        ai_markers = [
            '.claude',
            '.cursor',
            '.copilot',
            '.tabnine',
            '.codewhisperer',
            '.codesnippets',
            '.kite',
            '.ai',
            '.openai',
            '.aicode',
        ]
        
        # Create set of existing results for fast lookup
        existing_set = set()
        if existing_data:
            for marker, hits in existing_data.items():
                for hit in hits:
                    existing_set.add((marker, hit['repo_name'], hit['file_path']))
        
        results = {}
        for marker in ai_markers:
            # Build the search query for the file path
            query = f'path:{marker}'
            if min_stars > 0:
                query += f' stars:>={min_stars}'
            
            try:
                self.check_rate_limit()
                # Use the GitHub API to search for code files with the marker path
                code_results = self.github.search_code(query=query)
                marker_hits = []
                skipped_count = 0
                
                for file in code_results:
                    if len(marker_hits) >= max_repos_per_pattern:
                        break
                    
                    try:
                        repo = file.repository
                        # Check if this result already exists
                        result_key = (marker, repo.full_name, file.path)
                        if result_key in existing_set:
                            skipped_count += 1
                            continue  # Skip this result
                        
                        marker_hits.append({
                            'repo_name': repo.full_name,
                            'repo_url': repo.html_url,
                            'file_path': file.path,
                            'file_url': file.html_url,
                            'stars': repo.stargazers_count,
                            'description': repo.description,
                            'owner_type': repo.owner.type,
                            'owner_login': repo.owner.login,
                        })
                    except Exception as e:
                        logger.warning(f"Error processing file hit for {marker}: {e}")
                        continue
                
                results[marker] = marker_hits
                logger.info(f"Found {len(marker_hits)} new repos/files for marker {marker} (skipped {skipped_count} existing)")
            except Exception as e:
                logger.error(f"Error searching for marker {marker}: {e}")
                results[marker] = []
        return results

    def _scrape_marker_with_pagination(self, marker: str, max_repos: int, min_stars: int, 
                                      extract_contacts: bool, start_page: int, 
                                      start_position: int, state_manager: ScrapingStateManager) -> dict:
        """Scrape a specific marker with pagination support"""
        
        current_page = start_page
        current_position = start_position
        repos_found = 0
        total_repos_checked = 0
        skipped_count = 0
        new_repos_found = set()  # Track new repositories for auto-population
        
        # Build search query
        query = f'path:{marker}'
        if min_stars > 0:
            query += f' stars:>={min_stars}'
        
        logger.info(f"Starting paginated scraping for {marker} from page {start_page}, position {start_position}")
        
        while repos_found < max_repos:
            try:
                # Get search results for current page
                search_results = self._get_search_results_page(query, current_page)
                
                if not search_results:
                    logger.info(f"No more results for {marker} at page {current_page}")
                    break
                
                logger.info(f"Processing page {current_page} for {marker} ({len(search_results)} results)")
                
                # Process results starting from current position
                for i, file in enumerate(search_results):
                    if i < current_position:
                        continue  # Skip already processed results
                    
                    if repos_found >= max_repos:
                        break
                    
                    total_repos_checked += 1
                    
                    try:
                        repo = file.repository
                        repo_name = repo.full_name
                        
                        # Check if this repository already exists in database
                        if self._is_repo_in_database(repo_name):
                            skipped_count += 1
                            logger.debug(f"Skipping existing repository: {repo_name}")
                        else:
                            # Extract contact information for the repository owner (optional)
                            if extract_contacts:
                                owner_contacts = self.extract_contact_info(repo.owner.login)
                                
                                # If no contacts found in profile, try repository content
                                if owner_contacts['source'] == 'none':
                                    repo_contacts = self.extract_contacts_from_repo_content(repo.full_name)
                                    if repo_contacts['source'] != 'none':
                                        owner_contacts = repo_contacts
                            else:
                                owner_contacts = {'email': None, 'source': 'none'}
                            
                            # Get latest commit date for the repository (only if extract_contacts is True)
                            latest_commit_date = None
                            if extract_contacts:
                                latest_commit_date = self.get_latest_commit_date(repo.full_name)
                            
                            # Create new record
                            new_hit = MarkerHit(
                                marker=marker,
                                repo_name=repo.full_name,
                                repo_url=repo.html_url,
                                file_path=file.path,
                                file_url=file.html_url,
                                stars=repo.stargazers_count,
                                description=repo.description,
                                owner_type=repo.owner.type,
                                owner_login=repo.owner.login,
                                owner_email=owner_contacts['email'],
                                contact_source=owner_contacts['source'],
                                contact_extracted_at=datetime.utcnow().isoformat(),
                                latest_commit_date=latest_commit_date.isoformat() if latest_commit_date else None,
                                scraping_page=current_page,
                                scraping_position=i + 1
                            )
                            
                            # Add to database
                            with SessionLocal() as session:
                                try:
                                    session.add(new_hit)
                                    session.commit()
                                    repos_found += 1
                                    new_repos_found.add(repo_name)  # Track for auto-population
                                    logger.info(f"Added new repository: {marker} - {repo_name} (page {current_page}, position {i+1}) - {repos_found}/{max_repos}")
                                except IntegrityError:
                                    logger.debug(f"Repository already exists (race condition): {repo_name}")
                                    session.rollback()
                                    skipped_count += 1
                                except Exception as e:
                                    logger.error(f"Error adding repository {repo_name}: {e}")
                                    session.rollback()
                                    continue
                        
                        # Update state after each result (whether added or skipped)
                        current_position = i + 1
                        state_manager.update_scraping_state(marker, current_page, current_position, repo_name)
                        
                        # Rate limiting - increased delay
                        time.sleep(0.2)
                        
                    except Exception as e:
                        logger.warning(f"Error processing file hit for {marker}: {e}")
                        continue
                
                # Move to next page
                current_page += 1
                current_position = 0
                
                # Add delay between pages - increased
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error scraping {marker} at page {current_page}: {e}")
                break
        
        logger.info(f"Completed paginated scraping for {marker}: found {repos_found} new repos, checked {total_repos_checked}, skipped {skipped_count}")
        
        return {
            "repos_found": repos_found,
            "total_repos_checked": total_repos_checked,
            "skipped_count": skipped_count,
            "final_page": current_page,
            "final_position": current_position,
            "new_repos": list(new_repos_found) if 'new_repos_found' in locals() else []
        }

    def _should_skip_scraping_due_to_rate_limits(self) -> bool:
        """Check if we should skip scraping due to rate limits"""
        try:
            rate_limit = self.github.get_rate_limit()
            remaining = rate_limit.core.remaining
            limit = rate_limit.core.limit
            
            # If we have very few requests remaining, skip scraping
            if remaining < 200:
                logger.warning(f"Rate limit too low ({remaining}/{limit}). Skipping scraping to avoid rate limit errors.")
                return True
            
            return False
        except Exception as e:
            logger.warning(f"Error checking rate limit for skip decision: {e}")
            return False
    
    def _test_search_api(self) -> bool:
        """Test if the search API is working properly"""
        try:
            logger.info("Testing search API with a simple query...")
            
            # Test with a very common search term
            test_results = self._make_api_call_with_retry(
                self.github.search_code,
                query="README.md", per_page=1
            )
            
            if test_results and hasattr(test_results, 'totalCount'):
                logger.info(f"Search API test successful: {test_results.totalCount} results for README.md")
                return True
            elif test_results and len(test_results) > 0:
                logger.info(f"Search API test successful: {len(test_results)} results for README.md")
                return True
            else:
                logger.error("Search API test failed: No results returned")
                return False
                
        except Exception as e:
            logger.error(f"Search API test failed: {e}")
            return False

    def search_ai_code_generator_files_to_db(self, max_repos_per_pattern: int = 10, min_stars: int = 0, extract_contacts: bool = False) -> dict:
        """
        Search GitHub for repositories containing files that are markers for AI code generators
        and writes results directly to a SQLite database.
        Skips existing results to get new data points.

        Args:
            max_repos_per_pattern: Maximum NEW repositories to return per marker pattern.
                                  The scraper will continue searching until it finds exactly
                                  this many NEW repositories (not already in the database).
            min_stars: Minimum number of stars for repositories to include.
            extract_contacts: Whether to extract contact information (default: False for speed)
        """
        
        # Initialize state manager
        with SessionLocal() as db_session:
            state_manager = ScrapingStateManager(db_session)
        # List of AI code generator marker files to search for
        ai_markers = [
            '.claude',
            '.cursor',
            '.copilot',
            '.tabnine',
            '.codewhisperer',
            '.codesnippets',
            '.kite',
            '.ai',
            '.openai',
            '.aicode',
        ]
        
        # Reduce markers if rate limit is low (single token scenario)
        try:
            rate_limit = self.github.get_rate_limit()
            remaining = rate_limit.core.remaining
            
            if remaining < 1000:
                # Only process top 2 markers if rate limit is very low
                ai_markers = ai_markers[:2]
                logger.info(f"Rate limit very low ({remaining}), processing only top 2 markers: {ai_markers}")
            elif remaining < 2000:
                # Only process top 3 markers if rate limit is low
                ai_markers = ai_markers[:3]
                logger.info(f"Rate limit low ({remaining}), processing only top 3 markers: {ai_markers}")
            elif remaining < 3000:
                # Only process top 5 markers if rate limit is moderate
                ai_markers = ai_markers[:5]
                logger.info(f"Rate limit moderate ({remaining}), processing only top 5 markers: {ai_markers}")
        except Exception as e:
            logger.warning(f"Error checking rate limit for marker reduction: {e}")
        
        # Track new repositories found in this run to prevent duplicates within the same run
        new_repos_in_this_run = set()
        
        # Track new repositories found for auto-population of top contributors
        new_repositories_found = set()
        
        total_new_records = 0
        total_repos_found = 0
        total_skipped = 0
        
        # Check if we should skip scraping due to rate limits
        if self._should_skip_scraping_due_to_rate_limits():
            logger.warning("Skipping scraping due to low rate limit")
            return {
                "total_new_records": 0,
                "total_repos_found": 0,
                "total_repos_checked": 0,
                "new_repositories_found": 0,
                "summary": "Skipped scraping due to rate limits"
            }
        
        # Test search API before starting
        if not self._test_search_api():
            logger.error("Search API test failed. Skipping scraping.")
            return {
                "total_new_records": 0,
                "total_repos_found": 0,
                "total_repos_checked": 0,
                "new_repositories_found": 0,
                "summary": "Skipped scraping due to search API failure"
            }
        
        # Load existing repositories for duplicate checking
        logger.info("Loading existing repositories for duplicate checking...")
        existing_repos = set()
        
        # Use a separate session for loading existing repositories
        with SessionLocal() as load_session:
            try:
                # Load all existing repository names (distinct)
                existing_repos_list = load_session.query(MarkerHit.repo_name).distinct().all()
                existing_repos = {repo[0] for repo in existing_repos_list}
                
                logger.info(f"Loaded {len(existing_repos)} existing repositories for duplicate checking")
            except Exception as e:
                logger.error(f"Error loading existing repositories: {e}")
                # Continue with empty set if loading fails
                existing_repos = set()
        
        for marker in ai_markers:
            try:
                # Get current scraping state for this marker
                current_state = state_manager.get_scraping_state(marker)
                
                # Check if first result is already in database
                first_result = self._get_first_search_result(marker, min_stars)
                
                if first_result and self._is_repo_in_database(first_result['repo_name']):
                    # Resume from last known position
                    start_page = current_state['page']
                    start_position = current_state['position']
                    logger.info(f"Resuming {marker} scraping from page {start_page}, position {start_position}")
                else:
                    # Start fresh - new data available
                    start_page = 1
                    start_position = 0
                    logger.info(f"Starting fresh {marker} scraping")
                
                # Scrape with pagination
                result = self._scrape_marker_with_pagination(
                    marker, max_repos_per_pattern, min_stars, extract_contacts,
                    start_page, start_position, state_manager
                )
                
                # Update totals safely
                if result and isinstance(result, dict):
                    total_new_records += result.get('repos_found', 0)
                    total_repos_found += result.get('total_repos_checked', 0)
                    total_skipped += result.get('skipped_count', 0)
                    
                    # Update new repositories set for auto-population
                    new_repos_from_result = result.get('new_repos', [])
                    for repo_name in new_repos_from_result:
                        new_repositories_found.add(repo_name)
                        new_repos_in_this_run.add(repo_name)
                else:
                    logger.warning(f"No valid result returned for marker {marker}")
                
                # Add delay between markers - increased significantly
                time.sleep(10.0)
                    
            except Exception as e:
                logger.error(f"Error searching for marker {marker}: {e}")
                continue
        
        logger.info(f"=== SCRAPING SUMMARY ===")
        logger.info(f"Total repositories checked: {total_repos_found}")
        logger.info(f"New unique repositories found: {total_new_records}")
        logger.info(f"Repositories skipped (already in DB): {total_skipped}")
        # Calculate success rate safely to avoid division by zero
        total_checked = total_repos_found + total_skipped
        if total_checked > 0:
            success_rate = (total_new_records / total_checked) * 100
            logger.info(f"Success rate: {success_rate:.1f}% new repos found")
        else:
            logger.info("Success rate: N/A (no repositories checked)")
        
        # Auto-populate top contributors for new repositories
        if new_repositories_found and total_new_records > 0:
            logger.info(f"Auto-populating top contributors for {len(new_repositories_found)} new repositories...")
            try:
                self.auto_populate_top_contributors_for_new_repos(new_repositories_found)
            except Exception as e:
                logger.error(f"Error auto-populating top contributors: {e}")
                # Don't fail the entire scraping process if auto-population fails
        
        return {
            "total_new_records": total_new_records,
            "total_repos_found": total_repos_found,
            "total_repos_checked": total_repos_found + total_skipped,
            "new_repositories_found": len(new_repositories_found),
            "summary": f"Checked {total_repos_found + total_skipped} repositories, found {total_new_records} new unique ones"
        }

    def extract_contact_info(self, username: str) -> Dict[str, Optional[str]]:
        """
        Extract email information from a GitHub user's profile.
        
        Args:
            username: GitHub username
            
        Returns:
            Dictionary with email and source information
        """
        try:
            self.check_rate_limit()
            user = self.github.get_user(username)
            
            contacts = {
                'email': None,
                'source': 'none'
            }
            
            # Check public email
            if user.email:
                contacts['email'] = user.email
                contacts['source'] = 'github_profile'
            
            return contacts
            
        except Exception as e:
            logger.warning(f"Error extracting contact info for {username}: {e}")
            return {'email': None, 'source': 'none'}

    def get_latest_commit_date(self, repo_name: str) -> Optional[datetime]:
        """
        Get the latest commit date for a repository.
        
        Args:
            repo_name: Full repository name (owner/repo)
            
        Returns:
            datetime object of the latest commit, or None if not found
        """
        try:
            self.check_rate_limit()
            repo = self.github.get_repo(repo_name)
            commits = repo.get_commits()
            
            if commits.totalCount > 0:
                latest_commit = commits[0]  # First commit is the latest
                return latest_commit.commit.author.date
            else:
                return None
        except Exception as e:
            logger.warning(f"Error getting latest commit date for {repo_name}: {e}")
            return None

    def extract_contacts_from_repo_content(self, repo_name: str) -> Dict[str, Optional[str]]:
        """
        Extract contact information from repository content (README, package.json, etc.).
        
        Args:
            repo_name: Repository name (owner/repo)
            
        Returns:
            Dictionary with email and source information
        """
        try:
            owner, repo = repo_name.split('/', 1)
            repo_obj = self.github.get_repo(repo_name)
            
            files_to_check = ['README.md', 'package.json', 'CONTRIBUTING.md']
            
            for filename in files_to_check:
                try:
                    self.check_rate_limit()
                    content_obj = repo_obj.get_contents(filename)
                    content = content_obj.decoded_content.decode('utf-8', errors='ignore')
                    
                    # Extract emails
                    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
                    
                    if emails:
                        return {
                            'email': emails[0],
                            'source': 'repo_content'
                        }
                        
                except Exception as e:
                    continue  # File doesn't exist or other error
            
            return {'email': None, 'source': 'none'}
            
        except Exception as e:
            logger.warning(f"Error extracting contacts from repo {repo_name}: {e}")
            return {'email': None, 'source': 'none'}

    def check_database_integrity(self, database_type: str = "sqlite") -> dict:
        """
        Check database integrity and fix any issues like NULL IDs.
        
        Args:
            database_type: Type of database ("sqlite" or "postgresql")
            
        Returns:
            Dictionary with integrity check results
        """
        session = SessionLocal()
        
        try:
            # Check for NULL IDs
            null_id_count = session.query(MarkerHit).filter(MarkerHit.id.is_(None)).count()
            
            # Check for duplicate IDs
            duplicate_ids = session.query(MarkerHit.id, func.count(MarkerHit.id).label('count')).group_by(MarkerHit.id).having(func.count(MarkerHit.id) > 1).all()
            
            # Check for missing required fields
            missing_description = session.query(MarkerHit).filter(MarkerHit.description.is_(None)).count()
            
            issues = []
            fixes_applied = 0
            
            # Fix NULL IDs
            if null_id_count > 0:
                logger.warning(f"Found {null_id_count} records with NULL IDs")
                issues.append(f"NULL IDs: {null_id_count}")
                
                # Get the next available ID
                max_id = session.query(MarkerHit.id).order_by(MarkerHit.id.desc()).limit(1).scalar() or 0
                
                # Fix NULL IDs - use direct SQL update to avoid object issues
                try:
                    # Get all records with NULL IDs
                    null_records = session.query(MarkerHit).filter(MarkerHit.id.is_(None)).all()
                    
                    # Update each record with a new ID
                    for i, record in enumerate(null_records):
                        if record is not None:  # Check if record is not None
                            new_id = max_id + i + 1
                            try:
                                record.id = new_id
                                fixes_applied += 1
                            except Exception as e:
                                logger.error(f"Error updating record {i}: {e}")
                                continue
                    
                    session.commit()
                    logger.info(f"Fixed {fixes_applied} NULL IDs")
                    
                    # For PostgreSQL, reset the sequence if needed
                    if database_type == "postgresql" and fixes_applied > 0:
                        try:
                            # Reset the sequence to the new max ID
                            new_max_id = max_id + fixes_applied
                            session.execute(f"SELECT setval('marker_hits_id_seq', {new_max_id})")
                            session.commit()
                            logger.info(f"Reset PostgreSQL sequence to {new_max_id}")
                        except Exception as e:
                            logger.warning(f"Could not reset PostgreSQL sequence: {e}")
                            
                except Exception as e:
                    logger.error(f"Error fixing NULL IDs: {e}")
                    session.rollback()
            
            # Fix duplicate IDs
            if duplicate_ids:
                logger.warning(f"Found {len(duplicate_ids)} duplicate ID groups")
                issues.append(f"Duplicate IDs: {len(duplicate_ids)} groups")
                
                try:
                    for duplicate_id, count in duplicate_ids:
                        if count > 1:
                            # Get all records with this ID
                            records = session.query(MarkerHit).filter(MarkerHit.id == duplicate_id).all()
                            
                            # Keep the first one, update the rest
                            for i, record in enumerate(records[1:], 1):
                                if record is not None:  # Check if record is not None
                                    new_id = duplicate_id + i
                                    try:
                                        record.id = new_id
                                        fixes_applied += 1
                                    except Exception as e:
                                        logger.error(f"Error updating duplicate record: {e}")
                                        continue
                    
                    session.commit()
                    logger.info(f"Fixed {fixes_applied} duplicate IDs")
                except Exception as e:
                    logger.error(f"Error fixing duplicate IDs: {e}")
                    session.rollback()
            
            # Fix NULL descriptions
            if missing_description > 0:
                logger.warning(f"Found {missing_description} records with NULL descriptions")
                issues.append(f"NULL descriptions: {missing_description}")
                
                try:
                    session.query(MarkerHit).filter(MarkerHit.description.is_(None)).update({MarkerHit.description: ""})
                    session.commit()
                    logger.info(f"Fixed {missing_description} NULL descriptions")
                except Exception as e:
                    logger.error(f"Error fixing NULL descriptions: {e}")
                    session.rollback()
            
            total_records = session.query(MarkerHit).count()
            
            return {
                "total_records": total_records,
                "issues_found": issues,
                "fixes_applied": fixes_applied,
                "integrity_ok": len(issues) == 0,
                "database_type": database_type
            }
            
        except Exception as e:
            logger.error(f"Error during database integrity check: {e}")
            return {"error": str(e), "database_type": database_type}
        finally:
            session.close()

    def auto_populate_top_contributors_for_new_repos(self, new_repositories: set):
        """
        Auto-populate top contributor data for newly scraped repositories.
        
        Args:
            new_repositories: Set of repository names (owner/repo) to populate
        """
        if not new_repositories:
            logger.info("No new repositories to populate top contributors for")
            return
        
        logger.info(f"Starting auto-population of top contributors for {len(new_repositories)} new repositories...")
        
        populated_count = 0
        error_count = 0
        
        for repo_name in new_repositories:
            try:
                # Extract owner and repo from full name (e.g., "owner/repo")
                if '/' not in repo_name:
                    logger.warning(f"Invalid repo name format: {repo_name}")
                    continue
                    
                owner, repo = repo_name.split('/', 1)
                
                # Get contributors for the repository using GitHub API
                try:
                    self.check_rate_limit()
                    contributors = self.github.get_repo(f"{owner}/{repo}").get_contributors()
                    contributors_list = list(contributors)
                except Exception as e:
                    logger.warning(f"Failed to get contributors for {repo_name}: {e}")
                    error_count += 1
                    continue
                    
                if not contributors_list:
                    logger.info(f"No contributors found for {repo_name}")
                    continue
                    
                # Get the top contributor (first in the list)
                top_contributor = contributors_list[0]
                username = top_contributor.login
                
                if not username:
                    logger.warning(f"No username found for top contributor in {repo_name}")
                    continue
                    
                # Get the user's profile to extract email
                try:
                    self.check_rate_limit()
                    user = self.github.get_user(username)
                    email = user.email
                    
                    # Update all records for this repository with top contributor info
                    with SessionLocal() as session:
                        records = session.query(MarkerHit).filter(MarkerHit.repo_name == repo_name).all()
                        
                        for record in records:
                            record.top_contributor = username
                            record.top_contributor_email = email
                        
                        session.commit()
                        populated_count += 1
                        
                        if email:
                            logger.info(f"Updated {len(records)} records for {repo_name}: {username} ({email})")
                        else:
                            logger.info(f"Updated {len(records)} records for {repo_name}: {username} (no email)")
                            
                except Exception as e:
                    logger.warning(f"Failed to get user profile for {username}: {e}")
                    # Still update with username even if email fails
                    try:
                        with SessionLocal() as session:
                            records = session.query(MarkerHit).filter(MarkerHit.repo_name == repo_name).all()
                            
                            for record in records:
                                record.top_contributor = username
                                record.top_contributor_email = None
                            
                            session.commit()
                            populated_count += 1
                            logger.info(f"Updated {len(records)} records for {repo_name}: {username} (no email)")
                    except Exception as db_error:
                        logger.error(f"Failed to update database for {repo_name}: {db_error}")
                        error_count += 1
                        continue
                
                # Add delay to respect rate limits
                time.sleep(0.1)  # 100ms delay between requests
                
            except Exception as e:
                logger.error(f"Error processing top contributor for {repo_name}: {e}")
                error_count += 1
                continue
        
        logger.info(f"Auto-population complete: {populated_count} repositories updated, {error_count} errors")

class APICodeParser:
    """
    Parses code files from API content.
    Provides methods to extract comments, identifiers, and logic blocks from code.
    """
    
    LANGUAGE_EXTENSIONS = {
        'python': ['.py'],
        'javascript': ['.js', '.jsx', '.ts', '.tsx'],
        'java': ['.java'],
        'cpp': ['.cpp', '.cc', '.cxx', '.hpp', '.h'],
        'c': ['.c', '.h'],
        'csharp': ['.cs'],
        'go': ['.go'],
        'rust': ['.rs'],
        'php': ['.php'],
        'ruby': ['.rb'],
        'swift': ['.swift'],
        'kotlin': ['.kt'],
        'scala': ['.scala'],
        'r': ['.r', '.R'],
        'sql': ['.sql'],
        'shell': ['.sh', '.bash'],
        'powershell': ['.ps1'],
        'lua': ['.lua'],
        'dart': ['.dart']
    }
    
    COMMENT_PATTERNS = {
        'python': [r'#.*$', r'"""[\s\S]*?"""', r"'''[\s\S]*?'''"] ,
        'javascript': [r'//.*$', r'/\*[\s\S]*?\*/'],
        'java': [r'//.*$', r'/\*[\s\S]*?\*/'],
        'cpp': [r'//.*$', r'/\*[\s\S]*?\*/'],
        'c': [r'//.*$', r'/\*[\s\S]*?\*/'],
        'csharp': [r'//.*$', r'/\*[\s\S]*?\*/'],
        'go': [r'//.*$', r'/\*[\s\S]*?\*/'],
        'rust': [r'//.*$', r'/\*[\s\S]*?\*/'],
        'php': [r'//.*$', r'#.*$', r'/\*[\s\S]*?\*/'],
        'ruby': [r'#.*$', r'=begin[\s\S]*?=end'],
        'swift': [r'//.*$', r'/\*[\s\S]*?\*/'],
        'kotlin': [r'//.*$', r'/\*[\s\S]*?\*/'],
        'scala': [r'//.*$', r'/\*[\s\S]*?\*/'],
        'sql': [r'--.*$', r'/\*[\s\S]*?\*/'],
        'shell': [r'#.*$'],
        'lua': [r'--.*$', r'--\[\[[\s\S]*?\]\]']
    }
    
    def get_language_from_extension(self, file_path: str) -> Optional[str]:
        """Determine programming language from file extension."""
        ext = re.sub(r'\.\w+$', '', file_path).lower() # Remove .extension
        for lang, extensions in self.LANGUAGE_EXTENSIONS.items():
            if ext in extensions:
                return lang
        return None
    
    def extract_comments(self, content: str, language: str) -> List[str]:
        """Extract comments from code content using regex patterns."""
        comments = []
        patterns = self.COMMENT_PATTERNS.get(language, [])
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            comments.extend(matches)
        
        return comments
    
    def extract_identifiers(self, content: str, language: str) -> List[str]:
        """Extract identifiers (variable names, function names, etc.) from code."""
        identifiers = []
        
        if language == 'python':
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Name, ast.FunctionDef, ast.ClassDef)):
                        if hasattr(node, 'id'):
                            identifiers.append(node.id)
                        elif hasattr(node, 'name'):
                            identifiers.append(node.name)
            except:
                # Fallback to regex if AST parsing fails
                identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', content)
        else:
            # Generic identifier extraction for other languages
            identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', content)
        
        # Filter out common keywords and short identifiers
        keywords = {'if', 'else', 'for', 'while', 'do', 'try', 'catch', 'finally',
                   'return', 'break', 'continue', 'class', 'def', 'function',
                   'var', 'let', 'const', 'public', 'private', 'protected'}
        
        identifiers = [id for id in identifiers if len(id) > 1 and id.lower() not in keywords]
        
        return identifiers
    
    def extract_logic_blocks(self, content: str, language: str) -> List[str]:
        """Extract logic blocks (functions, classes, etc.) from code."""
        blocks = []
        
        if language == 'python':
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.If, ast.For, ast.While)):
                        blocks.append(type(node).__name__)
            except:
                pass
        
        # Generic patterns for other languages
        patterns = {
            'javascript': [r'function\s+\w+', r'class\s+\w+', r'if\s*\(', r'for\s*\(', r'while\s*\('],
            'java': [r'public\s+\w+', r'private\s+\w+', r'class\s+\w+', r'if\s*\(', r'for\s*\(', r'while\s*\('],
            'cpp': [r'\w+\s*\([^)]*\)\s*{', r'class\s+\w+', r'if\s*\(', r'for\s*\(', r'while\s*\('],
            'go': [r'func\s+\w+', r'if\s+', r'for\s+', r'switch\s+'],
        }
        
        if language in patterns:
            for pattern in patterns[language]:
                blocks.extend(re.findall(pattern, content))
        
        return blocks

class APICodeAnalyzer:
    """
    Analyzes code for various quality metrics and suspicious patterns.
    Provides methods to calculate comment density, naming entropy, repetition, complexity, and detect suspicious code.
    """
    
    SUSPICIOUS_PATTERNS = [
        r'eval\s*\(',  # Dynamic code execution
        r'exec\s*\(',  # Dynamic code execution
        r'subprocess\.',  # System calls
        r'os\.system',  # System calls
        r'__import__',  # Dynamic imports
        r'base64\.decode',  # Base64 decoding (potential obfuscation)
        r'[a-zA-Z0-9+/]{30,}=*',  # Base64-like strings
        r'\\x[0-9a-fA-F]{2}',  # Hex encoding
        r'chr\(\d+\)',  # Character encoding
        r'\.encode\(',  # Encoding operations
        r'crypt|cipher|decrypt',  # Cryptographic operations
        r'password|secret|key|token|api_key',  # Sensitive data
        r'backdoor|malware|exploit|payload',  # Malicious indicators
        r'keylog|steal|harvest',  # Suspicious activities
        r'shell|cmd|command',  # Shell execution
        r'socket|connect|bind|listen',  # Network operations
        r'urllib|requests|http',  # HTTP requests (context dependent)
        r'pickle|marshal|dill',  # Serialization (potential security risk)
        r'tempfile|mktemp',  # Temporary file operations
        r'random|uuid|guid',  # Random generation (context dependent)
    ]
    
    def calculate_comment_density(self, content: str, comments: List[str]) -> float:
        """Calculate the ratio of comment lines to total lines in the file."""
        total_lines = len(content.splitlines())
        comment_lines = sum(len(comment.splitlines()) for comment in comments)
        
        if total_lines == 0:
            return 0.0
        
        return comment_lines / total_lines
    
    def calculate_naming_entropy(self, identifiers: List[str]) -> float:
        """Calculate entropy of identifier names (higher = more random/suspicious)."""
        if not identifiers:
            return 0.0
        
        # Calculate character frequency
        all_chars = ''.join(identifiers).lower()
        char_counts = Counter(all_chars)
        total_chars = len(all_chars)
        
        if total_chars == 0:
            return 0.0
        
        # Calculate entropy
        entropy = 0.0
        for count in char_counts.values():
            probability = count / total_chars
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def calculate_pattern_repetition(self, content: str) -> float:
        """Calculate how repetitive the code patterns are (duplicate lines, etc)."""
        lines = content.splitlines()
        if len(lines) < 2:
            return 0.0
        
        # Remove whitespace and comments for pattern analysis
        cleaned_lines = []
        for line in lines:
            cleaned = re.sub(r'#.*$', '', line.strip())  # Remove comments
            cleaned = re.sub(r'//.*$', '', cleaned)  # Remove C-style comments
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
            if cleaned and len(cleaned) > 3:  # Skip very short lines
                cleaned_lines.append(cleaned)
        
        if len(cleaned_lines) < 2:
            return 0.0
        
        # Count duplicate lines
        line_counts = Counter(cleaned_lines)
        duplicates = sum(count - 1 for count in line_counts.values() if count > 1)
        
        return duplicates / len(cleaned_lines)
    
    def calculate_complexity_score(self, logic_blocks: List[str]) -> float:
        """Calculate complexity based on control structures and logic blocks."""
        complexity_weights = {
            'If': 1,
            'For': 2,
            'While': 2,
            'FunctionDef': 1,
            'ClassDef': 1,
            'function': 1,
            'class': 1,
            'if': 1,
            'for': 2,
            'while': 2,
        }
        
        total_complexity = sum(complexity_weights.get(block, 1) for block in logic_blocks)
        return min(total_complexity / 10.0, 1.0)  # Normalize to 0-1
    
    def find_suspicious_patterns(self, content: str) -> List[str]:
        """Find potentially suspicious patterns in code using regexes."""
        found_patterns = []
        
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                found_patterns.append(pattern)
        
        return found_patterns
    
    def analyze_file_content(self, content: str, file_info: Dict, repo_name: str, parser: APICodeParser) -> Optional[FileAnalysis]:
        """
        Analyze file content and return analysis results as a FileAnalysis object.
        Returns None if the file is not a supported language.
        """
        try:
            language = parser.get_language_from_extension(file_info['path'])
            if not language:
                return None
            
            # Extract code elements
            comments = parser.extract_comments(content, language)
            identifiers = parser.extract_identifiers(content, language)
            logic_blocks = parser.extract_logic_blocks(content, language)
            
            # Calculate metrics
            comment_density = self.calculate_comment_density(content, comments)
            naming_entropy = self.calculate_naming_entropy(identifiers)
            pattern_repetition = self.calculate_pattern_repetition(content)
            complexity = self.calculate_complexity_score(logic_blocks)
            suspicious_patterns = self.find_suspicious_patterns(content)
            
            # Calculate overall score (higher = more suspicious)
            overall_score = (
                (1 - comment_density) * 0.15 +  # Low comments = suspicious
                (naming_entropy / 5.0) * 0.25 +  # High entropy = suspicious
                pattern_repetition * 0.15 +  # High repetition = suspicious
                complexity * 0.15 +  # High complexity = suspicious
                (len(suspicious_patterns) / 10.0) * 0.3  # More patterns = suspicious
            )
            
            return FileAnalysis(
                repo_name=repo_name,
                file_path=file_info['path'],
                language=language,
                lines_of_code=len(content.splitlines()),
                file_size=file_info['size'],
                comment_density=comment_density,
                naming_entropy=naming_entropy,
                pattern_repetition=pattern_repetition,
                complexity_score=complexity,
                suspicious_patterns=suspicious_patterns,
                overall_score=min(overall_score, 1.0)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_info['path']}: {e}")
            return None

class APIDataAggregator:
    """
    Aggregates analysis results and exports data to JSON, CSV, and summary text files.
    """
    
    def __init__(self, output_dir: str = "api_analysis_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.file_analyses: List[FileAnalysis] = []
        self.repo_analyses: List[RepoAnalysis] = []
    
    def add_file_analysis(self, analysis: FileAnalysis):
        """Add a file analysis to the collection."""
        self.file_analyses.append(analysis)
    
    def aggregate_repo_analysis(self, repo_name: str, total_files_in_repo: int) -> RepoAnalysis:
        """
        Aggregate file analyses into repository-level analysis.
        Returns a RepoAnalysis object.
        """
        repo_files = [a for a in self.file_analyses if a.repo_name == repo_name]
        
        if not repo_files:
            return RepoAnalysis(repo_name, 0, total_files_in_repo, {}, 0, 0, 0, 0, [], 0, 0)
        
        # Calculate aggregated metrics
        languages = Counter(f.language for f in repo_files)
        avg_comment_density = sum(f.comment_density for f in repo_files) / len(repo_files)
        avg_naming_entropy = sum(f.naming_entropy for f in repo_files) / len(repo_files)
        avg_pattern_repetition = sum(f.pattern_repetition for f in repo_files) / len(repo_files)
        avg_complexity = sum(f.complexity_score for f in repo_files) / len(repo_files)
        
        # Find suspicious files (top 20% by score or score > 0.7)
        sorted_files = sorted(repo_files, key=lambda x: x.overall_score, reverse=True)
        suspicious_threshold = 0.7
        suspicious_files = [f.file_path for f in sorted_files if f.overall_score > suspicious_threshold]
        
        # If no files above threshold, take top 20%
        if not suspicious_files:
            suspicious_count = max(1, len(sorted_files) // 5)
            suspicious_files = [f.file_path for f in sorted_files[:suspicious_count]]
        
        overall_score = sum(f.overall_score for f in repo_files) / len(repo_files)
        analysis_coverage = len(repo_files) / total_files_in_repo if total_files_in_repo > 0 else 0
        
        repo_analysis = RepoAnalysis(
            repo_name=repo_name,
            total_files_analyzed=len(repo_files),
            total_files_in_repo=total_files_in_repo,
            languages=dict(languages),
            avg_comment_density=avg_comment_density,
            avg_naming_entropy=avg_naming_entropy,
            avg_pattern_repetition=avg_pattern_repetition,
            avg_complexity=avg_complexity,
            suspicious_files=suspicious_files,
            overall_score=overall_score,
            analysis_coverage=analysis_coverage
        )
        
        self.repo_analyses.append(repo_analysis)
        return repo_analysis
    
    def export_json(self, filename: str = "api_analysis_results.json"):
        """
        Export results to a JSON file (includes all file and repo analyses).
        """
        output_file = self.output_dir / filename
        
        data = {
            'metadata': {
                'analysis_type': 'github_api_scraping',
                'total_repos': len(self.repo_analyses),
                'total_files': len(self.file_analyses),
                'timestamp': str(datetime.now())
            },
            'file_analyses': [asdict(f) for f in self.file_analyses],
            'repo_analyses': [asdict(r) for r in self.repo_analyses]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"JSON results exported to {output_file}")
    
    def export_csv(self, filename: str = "api_analysis_results.csv"):
        """
        Export file analyses to a CSV file (spreadsheet format).
        """
        import csv
        output_file = self.output_dir / filename
        
        if self.file_analyses:
            # Convert dataclass objects to dictionaries
            data = [asdict(f) for f in self.file_analyses]
            
            # Write to CSV
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                if data:
                    fieldnames = data[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for row in data:
                        # Convert suspicious_patterns list to string
                        if 'suspicious_patterns' in row and isinstance(row['suspicious_patterns'], list):
                            row['suspicious_patterns'] = ', '.join(row['suspicious_patterns'])
                        writer.writerow(row)
            
            logger.info(f"CSV results exported to {output_file}")
    
    def generate_summary_report(self, filename: str = "api_analysis_summary.txt"):
        """
        Generate a human-readable summary analysis report (text file).
        """
        output_file = self.output_dir / filename
        
        with open(output_file, 'w') as f:
            f.write("GitHub API Code Quality Analysis Summary\n")
            f.write("=" * 45 + "\n\n")
            
            # Overall statistics
            total_files = len(self.file_analyses)
            high_risk_files = len([f for f in self.file_analyses if f.overall_score > 0.7])
            medium_risk_files = len([f for f in self.file_analyses if 0.3 <= f.overall_score <= 0.7])
            low_risk_files = total_files - high_risk_files - medium_risk_files
            
            f.write(f"Total Files Analyzed: {total_files}\n")
            f.write(f"High Risk Files (>0.7): {high_risk_files} ({high_risk_files/total_files*100:.1f}%)\n")
            f.write(f"Medium Risk Files (0.3-0.7): {medium_risk_files} ({medium_risk_files/total_files*100:.1f}%)\n")
            f.write(f"Low Risk Files (<0.3): {low_risk_files} ({low_risk_files/total_files*100:.1f}%)\n\n")
            
            # Language breakdown
            languages = Counter(f.language for f in self.file_analyses)
            f.write("Languages Analyzed:\n")
            for lang, count in languages.most_common():
                f.write(f"  {lang}: {count} files\n")
            f.write("\n")
            
            # Repository summaries
            for repo in sorted(self.repo_analyses, key=lambda x: x.overall_score, reverse=True):
                f.write(f"Repository: {repo.repo_name}\n")
                f.write(f"  Coverage: {repo.total_files_analyzed}/{repo.total_files_in_repo} files ({repo.analysis_coverage:.1%})\n")
                f.write(f"  Languages: {', '.join(f'{k}: {v}' for k, v in repo.languages.items())}\n")
                f.write(f"  Average Comment Density: {repo.avg_comment_density:.3f}\n")
                f.write(f"  Average Naming Entropy: {repo.avg_naming_entropy:.3f}\n")
                f.write(f"  Overall Risk Score: {repo.overall_score:.3f}\n")
                f.write(f"  Most Suspicious Files ({len(repo.suspicious_files)}):\n")
                for file in repo.suspicious_files[:3]:  # Show top 3
                    f.write(f"    - {file}\n")
                f.write("\n" + "-" * 40 + "\n\n")
        
        logger.info(f"Summary report generated: {output_file}")

class GitHubAPICodeAnalyzer:
    """
    Main orchestrator class for API-only analysis.
    Combines scraping, parsing, analysis, and aggregation/export.
    """
    
    def __init__(self, github_token: Optional[str] = None, output_dir: str = "api_analysis_output"):
        self.scraper = GitHubAPIScraper(github_token)
        self.parser = APICodeParser()
        self.analyzer = APICodeAnalyzer()
        self.aggregator = APIDataAggregator(output_dir)
    
    def analyze_repositories(self, 
                           search_query: str,
                           max_repos: int = 5,
                           max_files_per_repo: int = 50,
                           min_stars: int = 0,
                           file_size_limit_mb: float = 1.0):
        """
        Main method to analyze repositories via API only.
        Searches for repositories, analyzes code files, and exports results.
        """
        logger.info(f"Starting API-only analysis for query: '{search_query}'")
        logger.info(f"Settings: max_repos={max_repos}, max_files_per_repo={max_files_per_repo}")
        
        # Search for repositories
        repos = self.scraper.search_repositories(search_query, max_repos, min_stars)
        
        if not repos:
            logger.warning("No repositories found matching the search criteria")
            return
        
        total_files_analyzed = 0
        
        for i, repo_info in enumerate(repos, 1):
            logger.info(f"[{i}/{len(repos)}] Processing repository: {repo_info['name']}")
            
            try:
                repo = repo_info['repo_object']
                
                # Get repository contents through API
                contents = self.scraper.get_repository_contents(repo, max_files=max_files_per_repo)
                
                if not contents:
                    logger.warning(f"No code files found in {repo_info['name']}")
                    continue
                
                logger.info(f"  Found {len(contents)} code files")
                
                # Analyze files
                repo_files_analyzed = 0
                for j, file_info in enumerate(contents):
                    # Skip files that are too large
                    if file_info['size'] > file_size_limit_mb * 1024 * 1024:
                        logger.debug(f"  Skipping large file: {file_info['path']} ({file_info['size']} bytes)")
                        continue
                    
                    try:
                        # Get file content through API
                        content = self.scraper.get_file_content(file_info['content_object'])
                        
                        if content:
                            # Analyze the file
                            analysis = self.analyzer.analyze_file_content(
                                content, file_info, repo_info['name'], self.parser
                            )
                            
                            if analysis:
                                self.aggregator.add_file_analysis(analysis)
                                repo_files_analyzed += 1
                                total_files_analyzed += 1
                                
                                # Log suspicious files immediately
                                if analysis.overall_score > 0.7:
                                    logger.warning(f"    SUSPICIOUS: {analysis.file_path} (score: {analysis.overall_score:.3f})")
                        
                        # Rate limiting
                        time.sleep(self.scraper.rate_limit_delay)
                        
                    except Exception as e:
                        logger.warning(f"  Error analyzing file {file_info['path']}: {e}")
                        continue
                
                # Aggregate repository analysis
                total_files_in_repo = len(contents)  # This is our best estimate from API
                repo_analysis = self.aggregator.aggregate_repo_analysis(
                    repo_info['name'], total_files_in_repo
                )
                
                logger.info(f"  Analyzed {repo_files_analyzed} files from {repo_info['name']}")
                logger.info(f"  Repository risk score: {repo_analysis.overall_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error processing repository {repo_info['name']}: {e}")
                continue
        
        logger.info(f"Analysis complete! Total files analyzed: {total_files_analyzed}")
        
        # Export results
        self.aggregator.export_json()
        self.aggregator.export_csv()
        self.aggregator.generate_summary_report()
        
        # Print summary
        self.print_analysis_summary()
    
    def analyze_specific_repository(self, repo_url: str, max_files: int = 100):
        """
        Analyze a specific repository by URL (e.g., https://github.com/owner/repo).
        Exports results for that repository only.
        """
        try:
            # Extract owner/repo from URL
            if repo_url.endswith('.git'):
                repo_url = repo_url[:-4]
            
            parts = repo_url.replace('https://github.com/', '').split('/')
            if len(parts) < 2:
                logger.error("Invalid repository URL format")
                return
            
            repo_name = f"{parts[0]}/{parts[1]}"
            
            # Get repository object
            repo = self.scraper.github.get_repo(repo_name)
            
            logger.info(f"Analyzing specific repository: {repo_name}")
            
            # Get contents and analyze
            contents = self.scraper.get_repository_contents(repo, max_files=max_files)
            
            if not contents:
                logger.warning("No code files found in repository")
                return
            
            logger.info(f"Found {len(contents)} code files")
            
            files_analyzed = 0
            for file_info in contents:
                try:
                    content = self.scraper.get_file_content(file_info['content_object'])
                    
                    if content:
                        analysis = self.analyzer.analyze_file_content(
                            content, file_info, repo_name, self.parser
                        )
                        
                        if analysis:
                            self.aggregator.add_file_analysis(analysis)
                            files_analyzed += 1
                            
                            if analysis.overall_score > 0.7:
                                logger.warning(f"SUSPICIOUS: {analysis.file_path} (score: {analysis.overall_score:.3f})")
                    
                    time.sleep(self.scraper.rate_limit_delay)
                    
                except Exception as e:
                    logger.warning(f"Error analyzing file {file_info['path']}: {e}")
                    continue
            
            # Generate repository analysis
            repo_analysis = self.aggregator.aggregate_repo_analysis(repo_name, len(contents))
            
            logger.info(f"Analysis complete: {files_analyzed} files analyzed")
            logger.info(f"Repository risk score: {repo_analysis.overall_score:.3f}")
            
            # Export results
            self.aggregator.export_json(f"{repo_name.replace('/', '_')}_analysis.json")
            self.aggregator.export_csv(f"{repo_name.replace('/', '_')}_analysis.csv")
            self.aggregator.generate_summary_report(f"{repo_name.replace('/', '_')}_summary.txt")
            
        except Exception as e:
            logger.error(f"Error analyzing repository {repo_url}: {e}")
    
    def print_analysis_summary(self):
        """
        Print a summary of the analysis results to the console.
        Shows risk breakdown and top suspicious files.
        """
        if not self.aggregator.file_analyses:
            logger.info("No files were analyzed")
            return
        
        total_files = len(self.aggregator.file_analyses)
        high_risk = len([f for f in self.aggregator.file_analyses if f.overall_score > 0.7])
        medium_risk = len([f for f in self.aggregator.file_analyses if 0.3 <= f.overall_score <= 0.7])
        low_risk = total_files - high_risk - medium_risk
        
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total files analyzed: {total_files}")
        print(f"High risk files (>0.7): {high_risk} ({high_risk/total_files*100:.1f}%)")
        print(f"Medium risk files (0.3-0.7): {medium_risk} ({medium_risk/total_files*100:.1f}%)")
        print(f"Low risk files (<0.3): {low_risk} ({low_risk/total_files*100:.1f}%)")
        
        # Top suspicious files
        suspicious_files = sorted(
            [f for f in self.aggregator.file_analyses if f.overall_score > 0.5],
            key=lambda x: x.overall_score,
            reverse=True
        )[:5]
        
        if suspicious_files:
            print(f"\nTop {len(suspicious_files)} most suspicious files:")
            for f in suspicious_files:
                print(f"  {f.repo_name}/{f.file_path} (score: {f.overall_score:.3f})")
                if f.suspicious_patterns:
                    print(f"    Patterns: {', '.join(f.suspicious_patterns[:3])}")
        
        # Language breakdown
        languages = Counter(f.language for f in self.aggregator.file_analyses)
        print(f"\nLanguages analyzed: {', '.join(f'{k}: {v}' for k, v in languages.most_common())}")
        print("="*50)

def main():
    """
    Example usage of the API-only GitHub Code Analyzer.
    Run this file directly to perform a sample analysis.
    """
    # Initialize with GitHub token (recommended for higher rate limits)
    github_token = os.getenv('GITHUB_TOKEN')
    
    if not github_token:
        logger.warning("No GitHub token provided. You'll have lower rate limits.")
        logger.info("Set GITHUB_TOKEN environment variable for better performance.")
    
    analyzer = GitHubAPICodeAnalyzer(github_token)
    
    try:
        # Example 1: Search and analyze multiple repositories
        analyzer.analyze_repositories(
            search_query="python security tools",
            max_repos=3,
            max_files_per_repo=30,
            min_stars=10,  # Only repos with 10+ stars
            file_size_limit_mb=0.5  # Skip files larger than 500KB
        )
        
        # Example 2: Analyze a specific repository
        # analyzer.analyze_specific_repository(
        #     repo_url="https://github.com/owner/repository",
        #     max_files=50
        # )
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()