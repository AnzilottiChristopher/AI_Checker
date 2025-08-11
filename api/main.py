#!/usr/bin/env python3
"""
FastAPI application for AI Code Generator Marker API
"""
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import json
from typing import Optional, List
import sqlalchemy
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import Session, sessionmaker
import psycopg
import sys
import logging
import time
import asyncio
import signal
from contextlib import asynccontextmanager
import concurrent.futures

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_with_timeout(func, timeout_seconds=300, *args, **kwargs):
    """
    Run a function with a timeout to prevent infinite execution.
    Default timeout is 5 minutes (300 seconds).
    """
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            logger.error(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
            raise HTTPException(status_code=408, detail=f"Operation timed out after {timeout_seconds} seconds")
        except Exception as e:
            logger.error(f"Function {func.__name__} failed: {e}")
            raise e

# Import our scraper modules
from github_api_scraper import GitHubAPIScraper, init_database, MarkerHit

# Create FastAPI app
app = FastAPI(title="AI Code Generator Marker API", version="1.0.0")

# Add CORS middleware with proper security
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000", 
        "http://localhost:8080",
        "https://ai-checker-daxwyi63d-chris-anzilottis-projects.vercel.app",
        "https://ai-checker-3pgglrfai-chris-anzilottis-projects.vercel.app",
        "https://ai-checker-daxwyi63d-chris-anzilottis-projects.vercel.app",
        "https://*.vercel.app",
        "https://*.netlify.app",
        "*"  # Allow all origins for debugging - remove this in production
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "*"],
)

# Database setup
def get_db():
    """Database dependency"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")
    
    try:
        # Configure engine based on database type
        if database_url.startswith("postgresql"):
            # For Supabase, try connection pooling if available
            if "supabase.co" in database_url:
                # Use connection pooling for Supabase - replace the port with 6543
                if ":5432" in database_url:
                    pooled_url = database_url.replace(":5432", ":6543")
                else:
                    # If no port specified, add the pooled port
                    pooled_url = database_url.replace("supabase.co", "supabase.co:6543")
                engine = create_engine(
                    pooled_url.replace("postgresql://", "postgresql+psycopg://"), 
                    pool_pre_ping=True, 
                    pool_recycle=300,
                    pool_size=5,
                    max_overflow=10
                )
            else:
                # Regular PostgreSQL configuration with psycopg dialect
                engine = create_engine(
                    database_url.replace("postgresql://", "postgresql+psycopg://"), 
                    pool_pre_ping=True, 
                    pool_recycle=300,
                    pool_size=5,
                    max_overflow=10
                )
        else:
            # SQLite configuration (for local development)
            engine = create_engine(database_url, connect_args={"check_same_thread": False})
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    except Exception as e:
        logging.error(f"Database connection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint - basic status without database query"""
    try:
        database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "DATABASE_URL environment variable is not set",
                    "database": "unavailable"
                }
            )
        
        # Test database connection without querying tables (to avoid RLS issues)
        try:
            if database_url.startswith("postgresql"):
                # For Supabase, try connection pooling if available
                if "supabase.co" in database_url:
                    # Use connection pooling for Supabase - replace the port with 6543
                    if ":5432" in database_url:
                        pooled_url = database_url.replace(":5432", ":6543")
                    else:
                        # If no port specified, add the pooled port
                        pooled_url = database_url.replace("supabase.co", "supabase.co:6543")
                    engine = create_engine(
                        pooled_url.replace("postgresql://", "postgresql+psycopg://"), 
                        pool_pre_ping=True, 
                        pool_recycle=300,
                        pool_size=5,
                        max_overflow=10
                    )
                else:
                    # Regular PostgreSQL configuration with psycopg dialect
                    engine = create_engine(
                        database_url.replace("postgresql://", "postgresql+psycopg://"), 
                        pool_pre_ping=True, 
                        pool_recycle=300,
                        pool_size=5,
                        max_overflow=10
                    )
            else:
                # SQLite configuration (for local development)
                engine = create_engine(database_url, connect_args={"check_same_thread": False})
            
            # Just test the connection, don't query tables
            with engine.connect() as conn:
                # Simple connection test - this should work even with RLS
                conn.execute(text("SELECT 1"))
            
            return {
                "status": "healthy",
                "message": "API is running successfully",
                "database": "connected",
                "timestamp": "2025-08-07T20:25:00Z",
                "cors_enabled": True
            }
        except Exception as db_error:
            return {
                "status": "healthy",
                "message": "API is running successfully",
                "database": "connection_failed",
                "database_error": str(db_error),
                "timestamp": "2025-08-07T20:25:00Z",
                "cors_enabled": True
            }
                
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Health check error: {str(e)}",
                "database": "unavailable"
            }
        )

@app.get("/api/markers")
async def get_markers(db: Session = Depends(get_db)):
    """Get available markers from database"""
    try:
        # Get distinct marker values from the database
        markers = db.query(MarkerHit.marker).distinct().all()
        # Extract the marker values from the query results
        marker_list = [marker[0] for marker in markers if marker[0]]
        return {
            "markers": marker_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/owner_types")
async def get_owner_types(db: Session = Depends(get_db)):
    """Get available owner types from database"""
    try:
        # Get distinct owner_type values from the database
        owner_types = db.query(MarkerHit.owner_type).distinct().all()
        # Extract the owner_type values from the query results
        owner_type_list = [owner_type[0] for owner_type in owner_types if owner_type[0]]
        return {
            "owner_types": owner_type_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/owner_logins")
async def get_owner_logins(db: Session = Depends(get_db)):
    """Get available owner logins from database"""
    try:
        # Get distinct owner_login values from the database
        owner_logins = db.query(MarkerHit.owner_login).distinct().all()
        # Extract the owner_login values from the query results
        owner_login_list = [owner_login[0] for owner_login in owner_logins if owner_login[0]]
        return {
            "owner_logins": owner_login_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/contact-stats")
async def get_contact_stats(db: Session = Depends(get_db)):
    """Get contact statistics"""
    try:
        # Get total records
        total_records = db.query(MarkerHit).count()
        
        # Get records with email
        records_with_email = db.query(MarkerHit).filter(
            MarkerHit.owner_email.isnot(None),
            MarkerHit.owner_email != ""
        ).count()
        
        # Get records with any contact info
        records_with_contact = db.query(MarkerHit).filter(
            MarkerHit.owner_email.isnot(None) & (MarkerHit.owner_email != "")
        ).count()
        
        # Calculate percentages
        email_percentage = (records_with_email / total_records * 100) if total_records > 0 else 0
        contact_percentage = (records_with_contact / total_records * 100) if total_records > 0 else 0
        
        return {
            "total_records": total_records,
            "records_with_email": records_with_email,
            "records_with_contact": records_with_contact,
            "email_percentage": round(email_percentage, 2),
            "contact_percentage": round(contact_percentage, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/hits")
async def get_hits(
    limit: Optional[str] = Query(None),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("id"),
    sort_order: str = Query("desc"),
    filter_marker: Optional[str] = Query(None),
    marker: Optional[str] = Query(None),  # Frontend sends 'marker'
    filter_owner_type: Optional[str] = Query(None),
    owner_type: Optional[str] = Query(None),  # Frontend sends 'owner_type'
    filter_owner_login: Optional[str] = Query(None),
    has_email: Optional[bool] = Query(None),
    db: Session = Depends(get_db)
):
    """Get hits with filtering, sorting, and pagination"""
    try:
        # Start with base query
        query = db.query(MarkerHit)
        
        # Apply filters
        # Handle marker filter (frontend sends 'marker', backend expects 'filter_marker')
        marker_filter = filter_marker or marker
        if marker_filter:
            # Split by comma to handle multiple markers
            markers = [m.strip() for m in marker_filter.split(',') if m.strip()]
            if markers:
                # Create OR condition for multiple markers
                from sqlalchemy import or_
                marker_conditions = [MarkerHit.marker.contains(marker) for marker in markers]
                query = query.filter(or_(*marker_conditions))
        # Handle owner_type filter (frontend sends 'owner_type', backend expects 'filter_owner_type')
        owner_type_filter = filter_owner_type or owner_type
        if owner_type_filter:
            query = query.filter(MarkerHit.owner_type == owner_type_filter)
        if filter_owner_login:
            query = query.filter(MarkerHit.owner_login.contains(filter_owner_login))
        if has_email is not None:
            if has_email:
                query = query.filter(MarkerHit.owner_email.isnot(None), MarkerHit.owner_email != "")
            else:
                query = query.filter((MarkerHit.owner_email.is_(None)) | (MarkerHit.owner_email == ""))
        
        # Apply sorting - handle frontend's combined sort parameters
        if sort_by:
            # Map frontend sort parameters to database columns
            sort_mapping = {
                "stars_desc": ("stars", "desc"),
                "stars_asc": ("stars", "asc"),
                "commit_desc": ("latest_commit_date", "desc"),
                "commit_asc": ("latest_commit_date", "asc"),
                "name_desc": ("repo_name", "desc"),
                "name_asc": ("repo_name", "asc"),
                "id": ("id", "desc")
            }
            
            if sort_by in sort_mapping:
                column_name, order = sort_mapping[sort_by]
                sort_column = getattr(MarkerHit, column_name)
                
                # Special handling for commit date sorting to put "N/A" values at the bottom
                if column_name == "latest_commit_date":
                    if order.upper() == "DESC":
                        # For descending: put N/A, empty, and NULL values at the bottom
                        from sqlalchemy import case, desc, asc
                        query = query.order_by(
                            case(
                                (MarkerHit.latest_commit_date.is_(None), 0),
                                (MarkerHit.latest_commit_date == "", 0),
                                (MarkerHit.latest_commit_date == "N/A", 0),
                                else_=1
                            ).desc(),
                            MarkerHit.latest_commit_date.desc()
                        )
                    else:
                        # For ascending: put N/A, empty, and NULL values at the bottom
                        from sqlalchemy import case, desc, asc
                        query = query.order_by(
                            case(
                                (MarkerHit.latest_commit_date.is_(None), 0),
                                (MarkerHit.latest_commit_date == "", 0),
                                (MarkerHit.latest_commit_date == "N/A", 0),
                                else_=1
                            ).desc(),
                            MarkerHit.latest_commit_date.asc()
                        )
                else:
                    # Regular sorting for other columns
                    if order.upper() == "DESC":
                        query = query.order_by(sort_column.desc())
                    else:
                        query = query.order_by(sort_column.asc())
            else:
                # Fallback to direct column sorting
                if hasattr(MarkerHit, sort_by):
                    sort_column = getattr(MarkerHit, sort_by)
                    if sort_order.upper() == "DESC":
                        query = query.order_by(sort_column.desc())
                    else:
                        query = query.order_by(sort_column.asc())
        
        # Apply pagination
        query = query.offset(offset)
        
        # Handle limit - if empty string or None, don't limit (show all)
        # If limit is provided and valid, apply it; otherwise show all results
        if limit and limit.strip() and limit != "":
            try:
                limit_int = int(limit)
                if limit_int > 0:
                    query = query.limit(limit_int)
            except ValueError:
                # If limit is not a valid integer, don't apply limit (show all)
                pass
        # If no limit provided, show all results (no limit applied)
        
        # Execute query
        hits = query.all()
        
        # Convert to list of dictionaries
        result = []
        for hit in hits:
            result.append({
                "id": hit.id,
                "marker": hit.marker,
                "owner_type": hit.owner_type,
                "owner_login": hit.owner_login,
                "repo_name": hit.repo_name,
                "repo_url": hit.repo_url,
                "file_path": hit.file_path,
                "file_url": hit.file_url,
                "stars": hit.stars,
                "description": hit.description or "",
                "owner_email": hit.owner_email,
                "contact_source": hit.contact_source,
                "contact_extracted_at": str(hit.contact_extracted_at) if hit.contact_extracted_at else "",
                "latest_commit_date": str(hit.latest_commit_date) if hit.latest_commit_date else "",
                "top_contributor": hit.top_contributor,
                "top_contributor_email": hit.top_contributor_email
            })
        
        return {
            "hits": result,
            "total": len(result),
            "limit": limit if limit and limit.strip() else "all",
            "offset": offset
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/top-repos")
async def get_top_repos(
    limit: Optional[str] = Query(None),
    filter_marker: Optional[str] = Query(None),
    marker: Optional[str] = Query(None),  # Frontend sends 'marker'
    filter_owner_type: Optional[str] = Query(None),
    owner_type: Optional[str] = Query(None),  # Frontend sends 'owner_type'
    db: Session = Depends(get_db)
):
    """Get top repositories by stars"""
    try:
        # Start with base query
        query = db.query(MarkerHit)
        
        # Apply filters
        # Handle marker filter (frontend sends 'marker', backend expects 'filter_marker')
        marker_filter = filter_marker or marker
        if marker_filter:
            # Split by comma to handle multiple markers
            markers = [m.strip() for m in marker_filter.split(',') if m.strip()]
            if markers:
                # Create OR condition for multiple markers
                from sqlalchemy import or_
                marker_conditions = [MarkerHit.marker.contains(marker) for marker in markers]
                query = query.filter(or_(*marker_conditions))
        # Handle owner_type filter (frontend sends 'owner_type', backend expects 'filter_owner_type')
        owner_type_filter = filter_owner_type or owner_type
        if owner_type_filter:
            query = query.filter(MarkerHit.owner_type == owner_type_filter)
        
        # Sort by stars descending (top repositories)
        query = query.order_by(MarkerHit.stars.desc())
        
        # Apply limit - if empty string or None, don't limit (show all)
        # If limit is provided and valid, apply it; otherwise show all results
        if limit and limit.strip() and limit != "":
            try:
                limit_int = int(limit)
                if limit_int > 0:
                    query = query.limit(limit_int)
            except ValueError:
                # If limit is not a valid integer, don't apply limit (show all)
                pass
        # If no limit provided, show all results (no limit applied)
        
        # Execute query
        hits = query.all()
        
        # Convert to list of dictionaries
        result = []
        for hit in hits:
            result.append({
                "id": hit.id,
                "marker": hit.marker,
                "owner_type": hit.owner_type,
                "owner_login": hit.owner_login,
                "repo_name": hit.repo_name,
                "repo_url": hit.repo_url,
                "file_path": hit.file_path,
                "file_url": hit.file_url,
                "stars": hit.stars,
                "description": hit.description or "",
                "owner_email": hit.owner_email,
                "contact_source": hit.contact_source,
                "contact_extracted_at": str(hit.contact_extracted_at) if hit.contact_extracted_at else "",
                "latest_commit_date": str(hit.latest_commit_date) if hit.latest_commit_date else "",
                "top_contributor": hit.top_contributor,
                "top_contributor_email": hit.top_contributor_email
            })
        
        return {
            "hits": result,
            "total": len(result),
            "limit": limit if limit and limit.strip() else "all"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/api/run-scraper")
async def run_scraper(request: dict):
    """Run the scraper with support for multiple tokens"""
    try:
        github_token = request.get('github_token')
        backup_tokens = request.get('backup_tokens', [])  # List of backup tokens
        extract_contacts = request.get('extract_contacts', False)  # Default to False for speed
        max_repos_per_pattern = request.get('max_repos_per_pattern', 20)  # Increased from default 10
        
        if not github_token:
            raise HTTPException(status_code=400, detail="GitHub token is required")
        
        # Log token verification (first 8 chars only for security)
        logging.info(f"Scraper initialized with primary GitHub token (first 8 chars: {github_token[:8]}...)")
        if backup_tokens:
            logging.info(f"Backup tokens available: {len(backup_tokens)}")
            for i, token in enumerate(backup_tokens):
                logging.info(f"Backup token {i+1} (first 8 chars: {token[:8]}...)")
        
        # Initialize database
        try:
            init_database()
        except Exception as e:
            logging.error(f"Database initialization error: {e}")
            raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")
        
        # Create scraper instance with token rotation support
        try:
            scraper = GitHubAPIScraper(github_token, backup_tokens)
        except Exception as e:
            logging.error(f"Scraper initialization error: {e}")
            raise HTTPException(status_code=500, detail=f"Scraper initialization failed: {str(e)}")
        
        # Run scraper with optimized parameters
        try:
            def run_scraper_operation():
                return scraper.search_ai_code_generator_files_to_db(
                    max_repos_per_pattern=max_repos_per_pattern,
                    extract_contacts=extract_contacts
                )
            
            result = run_with_timeout(run_scraper_operation, timeout_seconds=480)  # 8 minute timeout (reduced from 10)
        except Exception as e:
            logging.error(f"Scraper execution error: {e}")
            raise HTTPException(status_code=500, detail=f"Scraper execution failed: {str(e)}")
        
        # Log token usage statistics
        if hasattr(scraper, 'rate_limit_errors') and scraper.rate_limit_errors:
            logging.info(f"Rate limit errors by token: {scraper.rate_limit_errors}")
        
        return {
            "status": "success",
            "message": result.get("summary", f"Scraper completed successfully. Added {result.get('total_new_records', 0)} new records."),
            "total_repos_found": result.get("total_repos_found", 0),
            "new_records": result.get("total_new_records", 0),
            "tokens_used": len(scraper.tokens) if hasattr(scraper, 'tokens') else 1,
            "rate_limit_errors": scraper.rate_limit_errors if hasattr(scraper, 'rate_limit_errors') else {}
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logging.error(f"Unexpected error in run_scraper: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected scraper error: {str(e)}")

@app.post("/api/update-commit-dates")
async def update_commit_dates(request: dict, db: Session = Depends(get_db)):
    """Update missing commit dates for existing records with support for multiple tokens"""
    try:
        github_token = request.get('github_token')
        backup_tokens = request.get('backup_tokens', [])  # List of backup tokens
        
        if not github_token:
            raise HTTPException(status_code=400, detail="GitHub token is required")
        
        # Log token verification (first 8 chars only for security)
        logging.info(f"Update commit dates initialized with primary GitHub token (first 8 chars: {github_token[:8]}...)")
        if backup_tokens:
            logging.info(f"Backup tokens available: {len(backup_tokens)}")
            for i, token in enumerate(backup_tokens):
                logging.info(f"Backup token {i+1} (first 8 chars: {token[:8]}...)")
        
        # Initialize database
        init_database()
        
        # Create scraper instance with token rotation support
        scraper = GitHubAPIScraper(github_token, backup_tokens)
        
        # Get records with missing commit dates (including "N/A" values)
        records_without_dates = db.query(MarkerHit).filter(
            (MarkerHit.latest_commit_date.is_(None)) | 
            (MarkerHit.latest_commit_date == "") |
            (MarkerHit.latest_commit_date == "N/A")
        ).all()
        
        updated_count = 0
        error_count = 0
        
        for record in records_without_dates:
            try:
                # Get latest commit date
                latest_commit_date = scraper.get_latest_commit_date(record.repo_name)
                
                if latest_commit_date:
                    # Update the record
                    record.latest_commit_date = latest_commit_date.isoformat()
                    updated_count += 1
                    
                    # Commit every 10 records to avoid long transactions
                    if updated_count % 10 == 0:
                        db.commit()
                        logging.info(f"Updated {updated_count} commit dates so far...")
                
                # Add delay to respect rate limits
                time.sleep(0.1)  # 100ms delay between requests
                
            except Exception as e:
                error_count += 1
                logging.warning(f"Error updating commit date for {record.repo_name}: {e}")
                continue
        
        # Final commit
        db.commit()
        
        return {
            "status": "success",
            "message": f"Updated {updated_count} commit dates. {error_count} errors encountered.",
            "updated_count": updated_count,
            "error_count": error_count,
            "total_records_processed": len(records_without_dates)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update error: {str(e)}")

@app.post("/api/test-github-token")
async def test_github_token(request: dict):
    """Test GitHub token authentication and rate limits with support for multiple tokens"""
    try:
        github_token = request.get('github_token')
        backup_tokens = request.get('backup_tokens', [])  # List of backup tokens
        
        if not github_token:
            raise HTTPException(status_code=400, detail="GitHub token is required")
        
        # Create scraper instance with token rotation support
        scraper = GitHubAPIScraper(github_token, backup_tokens)
        
        # Test rate limit to verify token is working
        try:
            rate_limit = scraper.github.get_rate_limit()
            remaining = rate_limit.core.remaining
            limit = rate_limit.core.limit
            
            # Test all tokens if backup tokens are provided
            token_results = []
            if backup_tokens:
                for i, token in enumerate(backup_tokens):
                    try:
                        test_scraper = GitHubAPIScraper(token)
                        test_rate_limit = test_scraper.github.get_rate_limit()
                        token_results.append({
                            "token_index": i + 1,
                            "rate_limit_remaining": test_rate_limit.core.remaining,
                            "rate_limit_total": test_rate_limit.core.limit,
                            "status": "working"
                        })
                    except Exception as e:
                        token_results.append({
                            "token_index": i + 1,
                            "status": "failed",
                            "error": str(e)
                        })
            
            return {
                "status": "success",
                "message": f"GitHub token is working! Rate limit: {remaining}/{limit} remaining",
                "rate_limit_remaining": remaining,
                "rate_limit_total": limit,
                "authenticated": True,
                "tokens_available": len(scraper.tokens) if hasattr(scraper, 'tokens') else 1,
                "backup_tokens_tested": token_results if backup_tokens else []
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"GitHub token test failed: {str(e)}",
                "authenticated": False
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token test error: {str(e)}")

@app.post("/api/run-scraper-fast")
async def run_scraper_fast(request: dict):
    """Run the scraper with high-throughput settings for maximum speed"""
    try:
        github_token = request.get('github_token')
        backup_tokens = request.get('backup_tokens', [])  # List of backup tokens
        
        if not github_token:
            raise HTTPException(status_code=400, detail="GitHub token is required")
        
        # Log token verification (first 8 chars only for security)
        logging.info(f"Fast scraper initialized with primary GitHub token (first 8 chars: {github_token[:8]}...)")
        if backup_tokens:
            logging.info(f"Backup tokens available: {len(backup_tokens)}")
            for i, token in enumerate(backup_tokens):
                logging.info(f"Backup token {i+1} (first 8 chars: {token[:8]}...)")
        
        # Initialize database
        try:
            init_database()
        except Exception as e:
            logging.error(f"Database initialization error: {e}")
            raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")
        
        # Create scraper instance with token rotation support
        try:
            scraper = GitHubAPIScraper(github_token, backup_tokens)
        except Exception as e:
            logging.error(f"Scraper initialization error: {e}")
            raise HTTPException(status_code=500, detail=f"Scraper initialization failed: {str(e)}")
        
        # Run scraper with high-throughput parameters
        try:
            def run_fast_scraper_operation():
                return scraper.search_ai_code_generator_files_to_db(
                    max_repos_per_pattern=100,  # Much higher limit for fast scraper
                    extract_contacts=False,  # Skip contact extraction for speed
                    min_stars=0  # Include all repos regardless of stars
                )
            
            result = run_with_timeout(run_fast_scraper_operation, timeout_seconds=600)  # 10 minute timeout for fast scraper (reduced from 15)
        except Exception as e:
            logging.error(f"Fast scraper execution error: {e}")
            raise HTTPException(status_code=500, detail=f"Fast scraper execution failed: {str(e)}")
        
        # Log token usage statistics
        if hasattr(scraper, 'rate_limit_errors') and scraper.rate_limit_errors:
            logging.info(f"Rate limit errors by token: {scraper.rate_limit_errors}")
        
        return {
            "status": "success",
            "message": result.get("summary", f"Fast scraper completed successfully. Added {result.get('total_new_records', 0)} new records."),
            "total_repos_found": result.get("total_repos_found", 0),
            "new_records": result.get("total_new_records", 0),
            "tokens_used": len(scraper.tokens) if hasattr(scraper, 'tokens') else 1,
            "rate_limit_errors": scraper.rate_limit_errors if hasattr(scraper, 'rate_limit_errors') else {}
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logging.error(f"Unexpected error in run_scraper_fast: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected fast scraper error: {str(e)}")

@app.get("/api/db-stats")
async def get_database_stats(db: Session = Depends(get_db)):
    """Get database statistics to understand data distribution"""
    try:
        # Get total count
        total_records = db.query(MarkerHit).count()
        
        # Get count by marker
        marker_counts = db.query(MarkerHit.marker, func.count(MarkerHit.id)).group_by(MarkerHit.marker).all()
        
        # Get count by owner type
        owner_type_counts = db.query(MarkerHit.owner_type, func.count(MarkerHit.id)).group_by(MarkerHit.owner_type).all()
        
        # Get recent records (last 24 hours)
        from datetime import datetime, timedelta
        yesterday = datetime.utcnow() - timedelta(days=1)
        recent_records = db.query(MarkerHit).filter(
            MarkerHit.contact_extracted_at >= yesterday.isoformat()
        ).count()
        
        return {
            "total_records": total_records,
            "marker_distribution": dict(marker_counts),
            "owner_type_distribution": dict(owner_type_counts),
            "recent_records_24h": recent_records,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail=f"Database stats error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AI Code Generator Marker API", "version": "1.0.0"}

@app.get("/api/test")
async def test_endpoint():
    """Simple test endpoint for debugging CORS"""
    return {
        "status": "ok",
        "message": "Test endpoint working",
        "timestamp": "2025-08-07T20:25:00Z"
    }

@app.get("/api/debug-db")
async def debug_database():
    """Debug endpoint to check database configuration"""
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        return {
            "error": "DATABASE_URL not set",
            "database_url": None
        }
    
    # Mask the password for security
    if database_url.startswith("postgresql://"):
        try:
            # Parse the URL to mask the password
            from urllib.parse import urlparse
            parsed = urlparse(database_url)
            masked_url = f"postgresql://{parsed.username}:***@{parsed.hostname}:{parsed.port}{parsed.path}"
        except:
            masked_url = "postgresql://***:***@***:***/***"
    else:
        masked_url = database_url
    
    return {
        "database_url_masked": masked_url,
        "database_type": "postgresql" if database_url.startswith("postgresql") else "sqlite",
        "has_supabase": "supabase.co" in database_url if database_url else False,
        "connection_test": "Try /api/health for connection test"
    }

@app.post("/api/migrate-database")
async def migrate_database_endpoint():
    """Run database migration to add new columns"""
    try:
        from github_api_scraper import migrate_database
        
        success = migrate_database()
        
        if success:
            return {
                "status": "success",
                "message": "Database migration completed successfully",
                "new_columns": ["top_contributor", "top_contributor_email"]
            }
        else:
            raise HTTPException(status_code=500, detail="Database migration failed")
            
    except Exception as e:
        logger.error(f"Migration endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Migration error: {str(e)}")
