#!/usr/bin/env python3
"""
GitHub API-Only Code Quality Scraper
Analyzes repositories through GitHub API without cloning to local filesystem.

This script provides classes and functions to search for repositories, fetch code files, analyze code quality and security, and export results, all using the GitHub API.

Main usage is via the GitHubAPICodeAnalyzer class. See api_usage_examples.py for example usage.
"""

import os
import json
import csv
import ast
import re
import math
import logging
import base64
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from datetime import datetime

import requests
from github import Github
from github.GithubException import RateLimitExceededException, UnknownObjectException
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database setup for direct SQLite writing
DATABASE_URL = "sqlite:///./ai_code_generator.db"
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
    owner_linkedin = Column(String)
    contact_source = Column(String)  # 'github_profile', 'repo_content', or 'none'
    contact_extracted_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

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
    
    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.github = Github(token) if token else Github()
        self.session = requests.Session()
        if token:
            self.session.headers.update({'Authorization': f'token {token}'})
        
        # Rate limiting
        self.requests_made = 0
        self.rate_limit_delay = 1.0  # seconds between requests
    
    def check_rate_limit(self):
        """Check and handle GitHub API rate limits. Sleeps if close to limit."""
        try:
            rate_limit = self.github.get_rate_limit()
            remaining = rate_limit.core.remaining
            
            if remaining < 100:  # Conservative threshold
                reset_time = rate_limit.core.reset.timestamp()
                current_time = time.time()
                sleep_time = max(reset_time - current_time, 60)
                
                logger.warning(f"Rate limit low ({remaining} remaining). Sleeping for {sleep_time:.0f} seconds...")
                time.sleep(sleep_time)
        except Exception as e:
            logger.warning(f"Error checking rate limit: {e}")
    
    def search_repositories(self, query: str, max_repos: int = 10, min_stars: int = 0) -> List[Dict]:
        """
        Search for repositories using GitHub API.
        Returns a list of repository metadata dicts, each including a repo object for further API calls.
        """
        try:
            # Add star filter to query if specified
            if min_stars > 0:
                query += f" stars:>={min_stars}"
                
            repos = self.github.search_repositories(query=query, sort="stars", order="desc")
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
        
        except UnknownObjectException:
            logger.warning(f"Path not found: {path}")
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
        
        return Path(filename).suffix.lower() in code_extensions

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

    def search_ai_code_generator_files_to_db(self, max_repos_per_pattern: int = 10, min_stars: int = 0, extract_contacts: bool = True) -> dict:
        """
        Search GitHub for repositories containing files that are markers for AI code generators
        and writes results directly to a SQLite database.
        Skips existing results to get new data points.

        Args:
            max_repos_per_pattern: Maximum repositories to return per marker pattern.
            min_stars: Minimum number of stars for repositories to include.
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
        
        # Get existing records from database for duplicate checking
        session = SessionLocal()
        existing_records = session.query(MarkerHit.marker, MarkerHit.repo_name, MarkerHit.file_path).all()
        existing_set = {(record[0], record[1], record[2]) for record in existing_records}
        session.close()
        
        total_new_records = 0
        for marker in ai_markers:
            # Build the search query for the file path
            query = f'path:{marker}'
            if min_stars > 0:
                query += f' stars:>={min_stars}'
            
            try:
                self.check_rate_limit()
                # Use the GitHub API to search for code files with the marker path
                code_results = self.github.search_code(query=query)
                
                processed_count = 0
                for file in code_results:
                    # Limit processing per marker to avoid taking too long
                    if processed_count >= max_repos_per_pattern:
                        logger.info(f"Reached limit of {max_repos_per_pattern} repos for marker {marker}")
                        break
                        
                    try:
                        repo = file.repository
                        # Check if this result already exists
                        result_key = (marker, repo.full_name, file.path)
                        if result_key in existing_set:
                            continue  # Skip this result
                        
                        # Extract contact information for the repository owner (optional)
                        if extract_contacts:
                            owner_contacts = self.extract_contact_info(repo.owner.login)
                            
                            # If no contacts found in profile, try repository content
                            if owner_contacts['source'] == 'none':
                                repo_contacts = self.extract_contacts_from_repo_content(repo.full_name)
                                if repo_contacts['source'] != 'none':
                                    owner_contacts = repo_contacts
                        else:
                            owner_contacts = {'email': None, 'linkedin_url': None, 'source': 'none'}
                        
                        session = SessionLocal()
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
                            owner_linkedin=owner_contacts['linkedin_url'],
                            contact_source=owner_contacts['source'],
                            contact_extracted_at=datetime.utcnow()
                        )
                        session.add(new_hit)
                        session.commit()
                        session.close()
                        total_new_records += 1
                        processed_count += 1
                        logger.info(f"Added new hit to DB: {marker} - {repo.full_name}/{file.path} (contacts: {owner_contacts['source']}) - {processed_count}/{max_repos_per_pattern}")
                    except Exception as e:
                        logger.warning(f"Error processing file hit for {marker}: {e}")
                        continue
                
                logger.info(f"Processed {processed_count} hits for marker {marker}")
            except Exception as e:
                logger.error(f"Error searching for marker {marker}: {e}")
        
        return {"total_new_records": total_new_records, "status": "completed"}

    def extract_contact_info(self, username: str) -> Dict[str, Optional[str]]:
        """
        Extract email and LinkedIn information from a GitHub user's profile.
        
        Args:
            username: GitHub username
            
        Returns:
            Dictionary with email, linkedin_url, and source information
        """
        try:
            self.check_rate_limit()
            user = self.github.get_user(username)
            
            contacts = {
                'email': None,
                'linkedin_url': None,
                'source': 'none'
            }
            
            # Check public email
            if user.email:
                contacts['email'] = user.email
                contacts['source'] = 'github_profile'
            
            # Check website for LinkedIn
            if user.blog:
                if 'linkedin.com' in user.blog:
                    contacts['linkedin_url'] = user.blog
                    if contacts['source'] == 'none':
                        contacts['source'] = 'github_profile'
            
            # Check bio for LinkedIn
            if user.bio:
                linkedin_match = re.search(r'linkedin\.com/in/[\w-]+', user.bio)
                if linkedin_match:
                    contacts['linkedin_url'] = f"https://{linkedin_match.group()}"
                    if contacts['source'] == 'none':
                        contacts['source'] = 'github_profile'
            
            return contacts
            
        except Exception as e:
            logger.warning(f"Error extracting contact info for {username}: {e}")
            return {'email': None, 'linkedin_url': None, 'source': 'none'}

    def extract_contacts_from_repo_content(self, repo_name: str) -> Dict[str, Optional[str]]:
        """
        Extract contact information from repository content (README, package.json, etc.).
        
        Args:
            repo_name: Repository name (owner/repo)
            
        Returns:
            Dictionary with email, linkedin_url, and source information
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
                    
                    # Extract LinkedIn URLs
                    linkedin_patterns = [
                        r'linkedin\.com/in/[\w-]+',
                        r'https?://[^\s]*linkedin\.com/in/[\w-]+'
                    ]
                    
                    linkedin_urls = []
                    for pattern in linkedin_patterns:
                        matches = re.findall(pattern, content)
                        linkedin_urls.extend(matches)
                    
                    if emails or linkedin_urls:
                        return {
                            'email': emails[0] if emails else None,
                            'linkedin_url': linkedin_urls[0] if linkedin_urls else None,
                            'source': 'repo_content'
                        }
                        
                except Exception as e:
                    continue  # File doesn't exist or other error
            
            return {'email': None, 'linkedin_url': None, 'source': 'none'}
            
        except Exception as e:
            logger.warning(f"Error extracting contacts from repo {repo_name}: {e}")
            return {'email': None, 'linkedin_url': None, 'source': 'none'}

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
        ext = Path(file_path).suffix.lower()
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
                'timestamp': str(pd.Timestamp.now())
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
        output_file = self.output_dir / filename
        
        if self.file_analyses:
            df = pd.DataFrame([asdict(f) for f in self.file_analyses])
            df['suspicious_patterns'] = df['suspicious_patterns'].apply(lambda x: ', '.join(x))
            df.to_csv(output_file, index=False)
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