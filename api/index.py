# backend.py
"""
FastAPI backend for storing and querying AI code generator marker hits.
- Uses SQLAlchemy with PostgreSQL for storage.
- Provides endpoints to query/filter results and run the scraper directly to database.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, DateTime, nullslast
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Base and MarkerHit from scraper to ensure consistency
from github_api_scraper import Base, MarkerHit, init_database

# Database setup - PostgreSQL
# Use environment variable for DATABASE_URL, fallback to SQLite for local development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ai_code_generator.db")

# Configure engine based on database type
if DATABASE_URL.startswith("postgresql"):
    # PostgreSQL configuration with improved connection pool settings
    engine = create_engine(
        DATABASE_URL, 
        pool_pre_ping=True, 
        pool_recycle=300,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30
    )
else:
    # SQLite configuration (for local development)
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Use MarkerHit model from github_api_scraper
# (Model definition removed to avoid conflicts)

# Pydantic model for API responses
class MarkerHitSchema(BaseModel):
    id: int
    marker: str
    repo_name: str
    repo_url: str
    file_path: str
    file_url: str
    stars: int
    description: Optional[str]
    owner_type: str
    owner_login: str
    owner_email: Optional[str]
    contact_source: Optional[str]
    contact_extracted_at: Optional[str]  # Changed from datetime to str to match database
    latest_commit_date: Optional[str]    # Changed from datetime to str to match database

    class Config:
        from_attributes = True

class ScraperRequest(BaseModel):
    github_token: Optional[str] = None
    extract_contacts: Optional[bool] = True

# Initialize database on startup (but don't fail if it doesn't work)
try:
    init_database()
except Exception as e:
    logger.warning(f"Database initialization failed (this is OK for development): {str(e)}")
    # Continue without database - the app will still work for basic endpoints

app = FastAPI(title="AI Code Generator Marker Backend")

# Global exception handler to sanitize error messages
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://localhost:8000",  # Local development
        "http://127.0.0.1:3000",  # Local development
        "http://127.0.0.1:8000",  # Local development
        "https://*.vercel.app",    # Vercel deployments
        "https://*.netlify.app",   # Netlify deployments (if you use it)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# TODO: For production deployment, replace with your specific frontend URL:
# allow_origins=["https://your-frontend-domain.vercel.app"]

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        # Test the connection
        db.execute("SELECT 1")
        yield db
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        db.close()
        raise HTTPException(status_code=500, detail="Database connection failed")
    finally:
        db.close()



@app.get("/hits", response_model=List[MarkerHitSchema])
def list_hits(
    marker: Optional[List[str]] = Query(None), 
    owner_type: Optional[str] = None, 
    owner_login: Optional[str] = None,
    has_email: Optional[str] = None,
    contact_source: Optional[str] = None,
    sort_by: Optional[str] = None,
    limit: Optional[int] = None
):
    """
    List all marker hits, with optional filtering and sorting.
    
    Parameters:
    - marker: Filter by marker type (e.g., '.claude', '.cursor') - can be a list for multiple markers
    - owner_type: Filter by owner type ('User' or 'Organization')
    - owner_login: Filter by specific owner username
    - has_email: Filter by whether email is available ('true' or 'false')
    - contact_source: Filter by contact source ('github_profile', 'repo_content', 'none')
    - sort_by: Sort order ('stars_desc', 'stars_asc', 'name_asc', 'name_desc')
    - limit: Maximum number of results to return
    """
    # Input validation
    if limit is not None and (limit < 1 or limit > 1000):
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 1000")
    
    if sort_by and sort_by not in ['stars_desc', 'stars_asc', 'name_asc', 'name_desc', 'commit_desc', 'commit_asc']:
        raise HTTPException(status_code=400, detail="Invalid sort_by parameter")
    
    if has_email and has_email.lower() not in ['true', 'false']:
        raise HTTPException(status_code=400, detail="has_email must be 'true' or 'false'")
    
    # Debug: Print received parameters
    print(f"DEBUG: Received marker parameter: {marker} (type: {type(marker)})")
    
    db = SessionLocal()
    try:
        query = db.query(MarkerHit)
        
        # Apply filters
        if marker:
            print(f"DEBUG: Applying marker filter: {marker}")
            if isinstance(marker, list) and len(marker) > 0:
                query = query.filter(MarkerHit.marker.in_(marker))
                print(f"DEBUG: Using IN filter with markers: {marker}")
            elif isinstance(marker, str):
                query = query.filter(MarkerHit.marker == marker)
                print(f"DEBUG: Using equality filter with marker: {marker}")
        if owner_type:
            query = query.filter(MarkerHit.owner_type == owner_type)
        if owner_login:
            query = query.filter(MarkerHit.owner_login == owner_login)
        if has_email:
            if has_email.lower() == 'true':
                query = query.filter(MarkerHit.owner_email.isnot(None))
            elif has_email.lower() == 'false':
                query = query.filter(MarkerHit.owner_email.is_(None))

        if contact_source:
            query = query.filter(MarkerHit.contact_source == contact_source)
        
        # Apply sorting
        if sort_by:
            if sort_by == 'stars_desc':
                query = query.order_by(MarkerHit.stars.desc())
            elif sort_by == 'stars_asc':
                query = query.order_by(MarkerHit.stars.asc())
            elif sort_by == 'commit_desc':
                query = query.order_by(MarkerHit.latest_commit_date.desc().nullslast())
            elif sort_by == 'commit_asc':
                query = query.order_by(MarkerHit.latest_commit_date.asc().nullslast())
            elif sort_by == 'name_asc':
                query = query.order_by(MarkerHit.repo_name.asc())
            elif sort_by == 'name_desc':
                query = query.order_by(MarkerHit.repo_name.desc())
            else:
                # Default to most recent first
                query = query.order_by(MarkerHit.id.desc())
        else:
            # Default to most recent first
            query = query.order_by(MarkerHit.id.desc())
        
        # Apply limit
        if limit:
            query = query.limit(limit)
        
        return query.all()
    finally:
        db.close()

@app.get("/markers", response_model=List[str])
def list_markers():
    """
    List all unique marker types in the database.
    """
    db = SessionLocal()
    try:
        markers = db.query(MarkerHit.marker).distinct().all()
        return [m[0] for m in markers]
    finally:
        db.close()

@app.get("/owner_types", response_model=List[str])
def list_owner_types():
    """
    List all unique owner types (User, Organization) in the database.
    """
    db = SessionLocal()
    try:
        owner_types = db.query(MarkerHit.owner_type).distinct().all()
        return [o[0] for o in owner_types if o[0]]
    finally:
        db.close()

@app.get("/owner_logins", response_model=List[str])
def list_owner_logins():
    """
    List all unique owner logins (user/org names) in the database.
    """
    db = SessionLocal()
    try:
        owner_logins = db.query(MarkerHit.owner_login).distinct().all()
        return [o[0] for o in owner_logins if o[0]]
    finally:
        db.close()

@app.get("/top-repos", response_model=List[MarkerHitSchema])
def get_top_repositories(
    limit: int = 10,
    marker: Optional[List[str]] = None,
    owner_type: Optional[str] = None
):
    """
    Get top repositories by star count.
    
    Parameters:
    - limit: Number of top repositories to return (default: 10)
    - marker: Filter by marker type (e.g., '.claude', '.cursor')
    - owner_type: Filter by owner type ('User' or 'Organization')
    """
    db = SessionLocal()
    try:
        query = db.query(MarkerHit)
        
        # Apply filters
        if marker:
            if isinstance(marker, list) and len(marker) > 0:
                query = query.filter(MarkerHit.marker.in_(marker))
            elif isinstance(marker, str):
                query = query.filter(MarkerHit.marker == marker)
        if owner_type:
            query = query.filter(MarkerHit.owner_type == owner_type)
        
        # Sort by stars descending and limit results
        query = query.order_by(MarkerHit.stars.desc()).limit(limit)
        
        return query.all()
    finally:
        db.close()

@app.get("/contacts")
def get_contacts(
    username: Optional[str] = None,
    has_email: Optional[bool] = None,

    contact_source: Optional[str] = None
):
    """
    Get contact information for users with optional filtering.
    """
    db = SessionLocal()
    try:
        query = db.query(MarkerHit)
        
        if username:
            query = query.filter(MarkerHit.owner_login == username)
        if has_email is not None:
            if has_email:
                query = query.filter(MarkerHit.owner_email.isnot(None))
            else:
                query = query.filter(MarkerHit.owner_email.is_(None))

        if contact_source:
            query = query.filter(MarkerHit.contact_source == contact_source)
        
        return query.all()
    finally:
        db.close()

@app.get("/contact-stats")
def get_contact_stats():
    """
    Get statistics about contact information collection.
    """
    db = SessionLocal()
    try:
        total_records = db.query(MarkerHit).count()
        records_with_email = db.query(MarkerHit).filter(MarkerHit.owner_email.isnot(None)).count()
        records_with_any_contact = db.query(MarkerHit).filter(
            MarkerHit.owner_email.isnot(None)
        ).count()
        
        return {
            "total_records": total_records,
            "records_with_email": records_with_email,
            "records_with_any_contact": records_with_any_contact,
            "email_percentage": round((records_with_email / total_records * 100), 2) if total_records > 0 else 0,
            "any_contact_percentage": round((records_with_any_contact / total_records * 100), 2) if total_records > 0 else 0
        }
    finally:
        db.close()

@app.get("/health")
def health_check():
    """Simple health check endpoint for the frontend to verify server status."""
    return {"status": "ok", "message": "Server is running", "database": "available" if engine else "unavailable"}

@app.post("/run-scraper")
async def run_scraper(request: ScraperRequest):
    """
    Run the GitHub scraper to collect new AI code generator marker data.
    This endpoint will:
    1. Run the scraper to find new repositories
    2. Write results directly to SQLite database (no JSON file)
    """
    try:
        # Check if GitHub token is available (from request or environment)
        github_token = request.github_token or os.getenv('GITHUB_TOKEN')
        if not github_token:
            raise HTTPException(
                status_code=400, 
                detail="GitHub token not found. Please set GITHUB_TOKEN environment variable or provide token in request."
            )
        
        # Import the scraper
        from github_api_scraper import GitHubAPIScraper
        
        # Initialize scraper with the provided token
        scraper = GitHubAPIScraper(token=github_token)
        
        # Run the scraper directly to database (no JSON file needed)
        results = scraper.search_ai_code_generator_files_to_db(
            max_repos_per_pattern=10,
            min_stars=0,
            extract_contacts=request.extract_contacts
        )
        
        return {
            "status": "success",
            "message": f"Scraper completed successfully! Added {results['total_new_records']} new records to database.",
            "new_records": results['total_new_records']
        }
        
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to import scraper module: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Scraper failed: {str(e)}"
        )

# Example usage:
# 1. Start the server: uvicorn backend:app --reload
# 2. Query: GET /hits, /hits?marker=.claude, /hits?owner_type=Organization, etc.
# 3. Run scraper: POST /run-scraper (writes directly to database) 

# Vercel handler for FastAPI
from mangum import Mangum

handler = Mangum(app) 