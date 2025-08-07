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
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
import psycopg2
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        "https://*.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Database setup
def get_db():
    """Database dependency"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")
    
    try:
        engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=300)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint with database connection test"""
    try:
        database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "DATABASE_URL environment variable is not set",
                    "database": "unavailable",
                    "debug_info": {
                        "database_url_exists": False,
                        "available_env_vars": list(os.environ.keys())
                    }
                }
            )
        
        # Test database connection
        engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=300)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM marker_hits"))
            count = result.scalar()
            
            return {
                "status": "healthy",
                "message": f"Database connection successful. Found {count} records.",
                "database": "connected",
                "record_count": count
            }
                
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Database error: {str(e)}",
                "database": "unavailable"
            }
        )

@app.get("/api/markers")
async def get_markers():
    """Get available markers"""
    return {
        "markers": [
            "ai-code-generator",
            "ai-code-gen", 
            "code-generator",
            "ai-generator"
        ]
    }

@app.get("/api/owner_types")
async def get_owner_types():
    """Get available owner types"""
    return {
        "owner_types": [
            "User",
            "Organization"
        ]
    }

@app.get("/api/owner_logins")
async def get_owner_logins():
    """Get available owner logins"""
    return {
        "owner_logins": [
            "example-user",
            "example-org"
        ]
    }

@app.get("/api/contact-stats")
async def get_contact_stats(db: Session = Depends(get_db)):
    """Get contact statistics"""
    try:
        # Get total records
        total_records = db.query(MarkerHit).count()
        
        # Get records with email
        records_with_email = db.query(MarkerHit).filter(
            MarkerHit.email.isnot(None),
            MarkerHit.email != ""
        ).count()
        
        # Get records with any contact info
        records_with_contact = db.query(MarkerHit).filter(
            (MarkerHit.email.isnot(None) & (MarkerHit.email != "")) |
            (MarkerHit.phone.isnot(None) & (MarkerHit.phone != ""))
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
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("id"),
    sort_order: str = Query("desc"),
    filter_marker: Optional[str] = Query(None),
    filter_owner_type: Optional[str] = Query(None),
    filter_owner_login: Optional[str] = Query(None),
    has_email: Optional[bool] = Query(None),
    db: Session = Depends(get_db)
):
    """Get hits with filtering, sorting, and pagination"""
    try:
        # Start with base query
        query = db.query(MarkerHit)
        
        # Apply filters
        if filter_marker:
            query = query.filter(MarkerHit.marker.contains(filter_marker))
        if filter_owner_type:
            query = query.filter(MarkerHit.owner_type == filter_owner_type)
        if filter_owner_login:
            query = query.filter(MarkerHit.owner_login.contains(filter_owner_login))
        if has_email is not None:
            if has_email:
                query = query.filter(MarkerHit.owner_email.isnot(None), MarkerHit.owner_email != "")
            else:
                query = query.filter((MarkerHit.owner_email.is_(None)) | (MarkerHit.owner_email == ""))
        
        # Apply sorting
        if hasattr(MarkerHit, sort_by):
            sort_column = getattr(MarkerHit, sort_by)
            if sort_order.upper() == "DESC":
                query = query.order_by(sort_column.desc())
            else:
                query = query.order_by(sort_column.asc())
        
        # Apply pagination
        query = query.offset(offset).limit(limit)
        
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
                "latest_commit_date": str(hit.latest_commit_date) if hit.latest_commit_date else ""
            })
        
        return {
            "hits": result,
            "total": len(result),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/api/run-scraper")
async def run_scraper(request: dict):
    """Run the scraper"""
    try:
        github_token = request.get('github_token')
        extract_contacts = request.get('extract_contacts', True)
        
        if not github_token:
            raise HTTPException(status_code=400, detail="GitHub token is required")
        
        # Initialize database
        init_database()
        
        # Create scraper instance
        scraper = GitHubAPIScraper(github_token)
        
        # Run scraper
        new_records = scraper.search_ai_code_generator_files_to_db(extract_contacts=extract_contacts)
        
        return {
            "status": "success",
            "message": f"Scraper completed successfully. Added {new_records} new records.",
            "new_records": new_records
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraper error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AI Code Generator Marker API", "version": "1.0.0"}
