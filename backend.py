# backend.py
"""
FastAPI backend for storing and querying AI code generator marker hits.
- Uses SQLAlchemy with SQLite for storage.
- Provides endpoints to query/filter results and run the scraper directly to database.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# Database setup
DATABASE_URL = "sqlite:///./ai_code_generator.db"
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

    class Config:
        orm_mode = True

class ScraperRequest(BaseModel):
    github_token: Optional[str] = None

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
def list_hits(marker: Optional[str] = None, owner_type: Optional[str] = None, owner_login: Optional[str] = None):
    """
    List all marker hits, with optional filtering by marker, owner_type, or owner_login.
    """
    db = next(get_db())
    query = db.query(MarkerHit)
    if marker:
        query = query.filter(MarkerHit.marker == marker)
    if owner_type:
        query = query.filter(MarkerHit.owner_type == owner_type)
    if owner_login:
        query = query.filter(MarkerHit.owner_login == owner_login)
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
        results = scraper.search_ai_code_generator_files_to_db()
        
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