#!/usr/bin/env python3
"""
Vercel serverless function handler
"""
import os
import sys
import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import sqlalchemy
from sqlalchemy import create_engine, text
import psycopg2

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)
        
        # Set CORS headers
        self.send_cors_headers()
        
        if path == "/api/health":
            self.handle_health()
        elif path == "/api/markers":
            self.handle_markers()
        elif path == "/api/owner_types":
            self.handle_owner_types()
        elif path == "/api/owner_logins":
            self.handle_owner_logins()
        elif path == "/api/contact-stats":
            self.handle_contact_stats()
        elif path == "/api/hits":
            self.handle_hits(query_params)
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        # Set CORS headers
        self.send_cors_headers()
        
        if path == "/api/run-scraper":
            self.handle_run_scraper()
        else:
            self.send_error(404, "Not Found")
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS preflight"""
        self.send_cors_headers()
        self.send_response(200)
        self.end_headers()
    
    def send_cors_headers(self):
        """Send CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    
    def handle_health(self):
        """Handle health check with database connection test"""
        try:
            # Get DATABASE_URL from environment
            database_url = os.getenv('DATABASE_URL')
            
            if not database_url:
                response = {
                    "status": "error",
                    "message": "DATABASE_URL not found in environment",
                    "database": "unavailable"
                }
                self.send_json_response(response, 500)
                return
            
            # Debug: Show what URL we're trying to connect to (masked)
            masked_url = database_url.replace(database_url.split('@')[0].split('://')[1], '***:***') if '@' in database_url else 'Unknown'
            
            # Create engine
            engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=300)
            
            # Test connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM marker_hits"))
                count = result.scalar()
                
                response = {
                    "status": "healthy",
                    "message": f"Database connection successful. Found {count} records.",
                    "database": "connected",
                    "record_count": count,
                    "connection_info": f"Connected to: {masked_url}"
                }
                self.send_json_response(response, 200)
                
        except Exception as e:
            response = {
                "status": "error",
                "message": f"Database error: {str(e)}",
                "database": "unavailable",
                "debug_info": {
                    "database_url_exists": bool(os.getenv('DATABASE_URL')),
                    "database_url_length": len(os.getenv('DATABASE_URL', '')),
                    "database_url_start": os.getenv('DATABASE_URL', '')[:20] + '...' if os.getenv('DATABASE_URL') else 'None'
                }
            }
            self.send_json_response(response, 500)
    
    def handle_markers(self):
        """Handle markers endpoint"""
        response = {
            "markers": [
                "ai-code-generator",
                "ai-code-gen",
                "code-generator",
                "ai-generator"
            ]
        }
        self.send_json_response(response, 200)
    
    def handle_owner_types(self):
        """Handle owner types endpoint"""
        response = {
            "owner_types": [
                "User",
                "Organization"
            ]
        }
        self.send_json_response(response, 200)
    
    def handle_owner_logins(self):
        """Handle owner logins endpoint"""
        response = {
            "owner_logins": [
                "example-user",
                "example-org"
            ]
        }
        self.send_json_response(response, 200)
    
    def handle_contact_stats(self):
        """Handle contact stats endpoint with real database query"""
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                response = {
                    "status": "error",
                    "message": "DATABASE_URL not found in environment"
                }
                self.send_json_response(response, 500)
                return
            
            engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=300)
            
            with engine.connect() as conn:
                # Get total records
                total_result = conn.execute(text("SELECT COUNT(*) FROM marker_hits"))
                total_records = total_result.scalar()
                
                # Get records with email
                email_result = conn.execute(text("SELECT COUNT(*) FROM marker_hits WHERE email IS NOT NULL AND email != ''"))
                records_with_email = email_result.scalar()
                
                # Get records with any contact info
                contact_result = conn.execute(text("SELECT COUNT(*) FROM marker_hits WHERE email IS NOT NULL AND email != '' OR phone IS NOT NULL AND phone != ''"))
                records_with_contact = contact_result.scalar()
                
                # Calculate percentages
                email_percentage = (records_with_email / total_records * 100) if total_records > 0 else 0
                contact_percentage = (records_with_contact / total_records * 100) if total_records > 0 else 0
                
                response = {
                    "total_records": total_records,
                    "records_with_email": records_with_email,
                    "records_with_contact": records_with_contact,
                    "email_percentage": round(email_percentage, 2),
                    "contact_percentage": round(contact_percentage, 2)
                }
                self.send_json_response(response, 200)
                
        except Exception as e:
            response = {
                "status": "error",
                "message": f"Database error: {str(e)}"
            }
            self.send_json_response(response, 500)
    
    def handle_hits(self, query_params):
        """Handle hits endpoint with filtering, sorting, and pagination"""
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                response = {
                    "status": "error",
                    "message": "DATABASE_URL not found in environment"
                }
                self.send_json_response(response, 500)
                return
            
            engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=300)
            
            # Parse query parameters
            limit = int(query_params.get('limit', [10])[0])
            offset = int(query_params.get('offset', [0])[0])
            sort_by = query_params.get('sort_by', ['id'])[0]
            sort_order = query_params.get('sort_order', ['desc'])[0]
            filter_marker = query_params.get('filter_marker', [''])[0]
            filter_owner_type = query_params.get('filter_owner_type', [''])[0]
            filter_owner_login = query_params.get('filter_owner_login', [''])[0]
            
            # Build WHERE clause
            where_conditions = []
            if filter_marker:
                where_conditions.append(f"marker LIKE '%{filter_marker}%'")
            if filter_owner_type:
                where_conditions.append(f"owner_type = '{filter_owner_type}'")
            if filter_owner_login:
                where_conditions.append(f"owner_login LIKE '%{filter_owner_login}%'")
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # Build query
            query = f"""
                SELECT 
                    id, marker, owner_type, owner_login, repo_name, file_path,
                    stars, forks, description, email, phone, contact_source,
                    contact_extracted_at, latest_commit_date
                FROM marker_hits 
                WHERE {where_clause}
                ORDER BY {sort_by} {sort_order.upper()}
                LIMIT {limit} OFFSET {offset}
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(query))
                rows = result.fetchall()
                
                # Convert to list of dictionaries, handling NULL values
                hits = []
                for row in rows:
                    hit = {
                        "id": row[0] if row[0] is not None else "",
                        "marker": row[1] if row[1] is not None else "",
                        "owner_type": row[2] if row[2] is not None else "",
                        "owner_login": row[3] if row[3] is not None else "",
                        "repo_name": row[4] if row[4] is not None else "",
                        "file_path": row[5] if row[5] is not None else "",
                        "stars": row[6] if row[6] is not None else "",
                        "forks": row[7] if row[7] is not None else "",
                        "description": row[8] if row[8] is not None else "",
                        "email": row[9] if row[9] is not None else "",
                        "phone": row[10] if row[10] is not None else "",
                        "contact_source": row[11] if row[11] is not None else "",
                        "contact_extracted_at": str(row[12]) if row[12] is not None else "",
                        "latest_commit_date": str(row[13]) if row[13] is not None else ""
                    }
                    hits.append(hit)
                
                response = {
                    "hits": hits,
                    "total": len(hits),
                    "limit": limit,
                    "offset": offset
                }
                self.send_json_response(response, 200)
                
        except Exception as e:
            response = {
                "status": "error",
                "message": f"Database error: {str(e)}"
            }
            self.send_json_response(response, 500)
    
    def handle_run_scraper(self):
        """Handle run scraper endpoint"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            github_token = data.get('github_token')
            extract_contacts = data.get('extract_contacts', True)
            
            if not github_token:
                response = {
                    "status": "error",
                    "message": "GitHub token is required"
                }
                self.send_json_response(response, 400)
                return
            
            # Import scraper modules
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from github_api_scraper import GitHubAPIScraper, init_database
            
            # Initialize database
            init_database()
            
            # Create scraper instance
            scraper = GitHubAPIScraper(github_token)
            
            # Run scraper
            new_records = scraper.search_ai_code_generator_files_to_db(extract_contacts=extract_contacts)
            
            response = {
                "status": "success",
                "message": f"Scraper completed successfully. Added {new_records} new records.",
                "new_records": new_records
            }
            self.send_json_response(response, 200)
            
        except Exception as e:
            response = {
                "status": "error",
                "message": f"Scraper error: {str(e)}"
            }
            self.send_json_response(response, 500)
    
    def send_json_response(self, data, status_code):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8')) 