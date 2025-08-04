# backend.py
"""
FastAPI backend for storing and querying AI code generator marker hits.
- Uses SQLAlchemy with PostgreSQL for storage.
- Provides endpoints to query/filter results and run the scraper directly to database.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, DateTime, nullslast
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import os

# Database setup - PostgreSQL
# Use environment variable for DATABASE_URL, fallback to SQLite for local development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ai_code_generator.db")

# Configure engine based on database type
if DATABASE_URL.startswith("postgresql"):
    # PostgreSQL configuration
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
else:
    # SQLite configuration (for local development)
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy model for a marker hit
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
    # Repository activity fields
    latest_commit_date = Column(DateTime)

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
    owner_linkedin: Optional[str]
    contact_source: Optional[str]
    contact_extracted_at: Optional[datetime]
    latest_commit_date: Optional[datetime]

    class Config:
        from_attributes = True

class ScraperRequest(BaseModel):
    github_token: Optional[str] = None
    extract_contacts: Optional[bool] = True

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Code Generator Marker Backend")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



@app.get("/hits", response_model=List[MarkerHitSchema])
def list_hits(
    marker: Optional[str] = None, 
    owner_type: Optional[str] = None, 
    owner_login: Optional[str] = None,
    has_email: Optional[str] = None,
    has_linkedin: Optional[str] = None,
    contact_source: Optional[str] = None,
    sort_by: Optional[str] = None,
    limit: Optional[int] = None
):
    """
    List all marker hits, with optional filtering and sorting.
    
    Parameters:
    - marker: Filter by marker type (e.g., '.claude', '.cursor')
    - owner_type: Filter by owner type ('User' or 'Organization')
    - owner_login: Filter by specific owner username
    - has_email: Filter by whether email is available ('true' or 'false')
    - has_linkedin: Filter by whether LinkedIn is available ('true' or 'false')
    - contact_source: Filter by contact source ('github_profile', 'repo_content', 'none')
    - sort_by: Sort order ('stars_desc', 'stars_asc', 'name_asc', 'name_desc')
    - limit: Maximum number of results to return
    """
    db = next(get_db())
    query = db.query(MarkerHit)
    
    # Apply filters
    if marker:
        query = query.filter(MarkerHit.marker == marker)
    if owner_type:
        query = query.filter(MarkerHit.owner_type == owner_type)
    if owner_login:
        query = query.filter(MarkerHit.owner_login == owner_login)
    if has_email:
        if has_email.lower() == 'true':
            query = query.filter(MarkerHit.owner_email.isnot(None))
        elif has_email.lower() == 'false':
            query = query.filter(MarkerHit.owner_email.is_(None))
    if has_linkedin:
        if has_linkedin.lower() == 'true':
            query = query.filter(MarkerHit.owner_linkedin.isnot(None))
        elif has_linkedin.lower() == 'false':
            query = query.filter(MarkerHit.owner_linkedin.is_(None))
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

@app.get("/markers", response_model=List[str])
def list_markers():
    """
    List all unique marker types in the database.
    """
    db = next(get_db())
    markers = db.query(MarkerHit.marker).distinct().all()
    return [m[0] for m in markers]

@app.get("/owner_types", response_model=List[str])
def list_owner_types():
    """
    List all unique owner types (User, Organization) in the database.
    """
    db = next(get_db())
    owner_types = db.query(MarkerHit.owner_type).distinct().all()
    return [o[0] for o in owner_types if o[0]]

@app.get("/owner_logins", response_model=List[str])
def list_owner_logins():
    """
    List all unique owner logins (user/org names) in the database.
    """
    db = next(get_db())
    owner_logins = db.query(MarkerHit.owner_login).distinct().all()
    return [o[0] for o in owner_logins if o[0]]

@app.get("/top-repos", response_model=List[MarkerHitSchema])
def get_top_repositories(
    limit: int = 10,
    marker: Optional[str] = None,
    owner_type: Optional[str] = None
):
    """
    Get top repositories by star count.
    
    Parameters:
    - limit: Number of top repositories to return (default: 10)
    - marker: Filter by marker type (e.g., '.claude', '.cursor')
    - owner_type: Filter by owner type ('User' or 'Organization')
    """
    db = next(get_db())
    query = db.query(MarkerHit)
    
    # Apply filters
    if marker:
        query = query.filter(MarkerHit.marker == marker)
    if owner_type:
        query = query.filter(MarkerHit.owner_type == owner_type)
    
    # Sort by stars descending and limit results
    query = query.order_by(MarkerHit.stars.desc()).limit(limit)
    
    return query.all()

@app.get("/contacts")
def get_contacts(
    username: Optional[str] = None,
    has_email: Optional[bool] = None,
    has_linkedin: Optional[bool] = None,
    contact_source: Optional[str] = None
):
    """
    Get contact information for users with optional filtering.
    """
    db = next(get_db())
    query = db.query(MarkerHit)
    
    if username:
        query = query.filter(MarkerHit.owner_login == username)
    if has_email is not None:
        if has_email:
            query = query.filter(MarkerHit.owner_email.isnot(None))
        else:
            query = query.filter(MarkerHit.owner_email.is_(None))
    if has_linkedin is not None:
        if has_linkedin:
            query = query.filter(MarkerHit.owner_linkedin.isnot(None))
        else:
            query = query.filter(MarkerHit.owner_linkedin.is_(None))
    if contact_source:
        query = query.filter(MarkerHit.contact_source == contact_source)
    
    return query.all()

@app.get("/contact-stats")
def get_contact_stats():
    """
    Get statistics about contact information collection.
    """
    db = next(get_db())
    
    total_records = db.query(MarkerHit).count()
    records_with_email = db.query(MarkerHit).filter(MarkerHit.owner_email.isnot(None)).count()
    records_with_linkedin = db.query(MarkerHit).filter(MarkerHit.owner_linkedin.isnot(None)).count()
    records_with_any_contact = db.query(MarkerHit).filter(
        (MarkerHit.owner_email.isnot(None)) | (MarkerHit.owner_linkedin.isnot(None))
    ).count()
    
    return {
        "total_records": total_records,
        "records_with_email": records_with_email,
        "records_with_linkedin": records_with_linkedin,
        "records_with_any_contact": records_with_any_contact,
        "email_percentage": round((records_with_email / total_records * 100), 2) if total_records > 0 else 0,
        "linkedin_percentage": round((records_with_linkedin / total_records * 100), 2) if total_records > 0 else 0,
        "any_contact_percentage": round((records_with_any_contact / total_records * 100), 2) if total_records > 0 else 0
    }

@app.get("/health")
def health_check():
    """Simple health check endpoint for the frontend to verify server status."""
    return {"status": "ok", "message": "Server is running"}

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