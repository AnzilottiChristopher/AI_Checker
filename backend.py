# backend.py
"""
FastAPI backend for storing and querying AI code generator marker hits.
- Uses SQLAlchemy with SQLite for storage.
- Provides endpoints to import data from JSON and to query/filter results.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
import json
from pathlib import Path

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

# Endpoint to import data from JSON file
def import_json_to_db(json_path: Path, db):
    """
    Import marker hits from a JSON file into the database.
    Expects the format produced by ai_code_generator_marker_search().
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    count = 0
    for marker, hits in data.items():
        for hit in hits:
            # Check if already exists (by marker, repo_name, file_path)
            exists = db.query(MarkerHit).filter_by(
                marker=marker,
                repo_name=hit['repo_name'],
                file_path=hit['file_path']
            ).first()
            if not exists:
                db.add(MarkerHit(
                    marker=marker,
                    repo_name=hit['repo_name'],
                    repo_url=hit['repo_url'],
                    file_path=hit['file_path'],
                    file_url=hit['file_url'],
                    stars=hit.get('stars', 0),
                    description=hit.get('description'),
                    owner_type=hit.get('owner_type', ''),
                    owner_login=hit.get('owner_login', ''),
                ))
                count += 1
    db.commit()
    return count

@app.post("/import-markers", response_class=JSONResponse)
def import_markers(json_file: Optional[str] = Query(None, description="Path to JSON file to import")):
    """
    Import marker hits from a JSON file (default: ai_code_generator_analysis/ai_code_generator_markers.json).
    """
    db = next(get_db())
    if json_file is None:
        json_path = Path("ai_code_generator_analysis/ai_code_generator_markers.json")
    else:
        json_path = Path(json_file)
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {json_path}")
    count = import_json_to_db(json_path, db)
    return {"imported": count}

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

# Example usage:
# 1. Start the server: uvicorn backend:app --reload
# 2. Import data: POST /import-markers (optionally with ?json_file=path)
# 3. Query: GET /hits, /hits?marker=.claude, /hits?owner_type=Organization, etc. 