from http.server import BaseHTTPRequestHandler
import json
import os
from urllib.parse import urlparse, parse_qs

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)
        
        if path == '/api/health':
            try:
                from sqlalchemy import create_engine, text
                import psycopg2
                
                database_url = os.getenv('DATABASE_URL')
                if not database_url:
                    response = {"status": "error", "message": "Database URL not configured", "database": "unavailable"}
                    self.wfile.write(json.dumps(response).encode())
                    return
                
                engine = create_engine(database_url)
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT COUNT(*) FROM marker_hits"))
                    count = result.scalar()
                    response = {"status": "ok", "message": "Server is running", "database": "connected", "record_count": count}
                    
            except Exception as e:
                response = {"status": "error", "message": f"Database error: {str(e)}", "database": "unavailable"}
                self.wfile.write(json.dumps(response).encode())
                return
        elif path == '/api/markers':
            response = [".claude", ".cursor", ".copilot"]
        elif path == '/api/owner_types':
            response = ["User", "Organization"]
        elif path == '/api/owner_logins':
            response = ["example-user", "example-org"]
        elif path == '/api/contact-stats':
            try:
                from sqlalchemy import create_engine, text
                from sqlalchemy.orm import sessionmaker
                import psycopg2
                
                database_url = os.getenv('DATABASE_URL')
                if not database_url:
                    response = {"error": "Database URL not configured"}
                    self.wfile.write(json.dumps(response).encode())
                    return
                
                engine = create_engine(database_url)
                SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
                
                with SessionLocal() as session:
                    total_result = session.execute(text("SELECT COUNT(*) FROM marker_hits"))
                    total_records = total_result.scalar()
                    
                    email_result = session.execute(text("SELECT COUNT(*) FROM marker_hits WHERE email IS NOT NULL AND email != ''"))
                    records_with_email = email_result.scalar()
                    
                    contact_result = session.execute(text("SELECT COUNT(*) FROM marker_hits WHERE email IS NOT NULL AND email != '' OR contact_source IS NOT NULL AND contact_source != ''"))
                    records_with_any_contact = contact_result.scalar()
                    
                    email_percentage = round((records_with_email / total_records * 100) if total_records > 0 else 0, 2)
                    any_contact_percentage = round((records_with_any_contact / total_records * 100) if total_records > 0 else 0, 2)
                    
                    response = {
                        "total_records": total_records,
                        "records_with_email": records_with_email,
                        "records_with_any_contact": records_with_any_contact,
                        "email_percentage": email_percentage,
                        "any_contact_percentage": any_contact_percentage
                    }
                    
            except Exception as e:
                response = {"error": f"Database error: {str(e)}"}
                self.wfile.write(json.dumps(response).encode())
                return
        elif path == '/api/hits':
            limit = int(query_params.get('limit', [10])[0])
            offset = int(query_params.get('offset', [0])[0])
            sort_by = query_params.get('sort_by', ['id'])[0]
            has_email = query_params.get('has_email', [None])[0]
            
            try:
                from sqlalchemy import create_engine, text
                from sqlalchemy.orm import sessionmaker
                import psycopg2
                
                database_url = os.getenv('DATABASE_URL')
                if not database_url:
                    response = {"error": "Database URL not configured"}
                    self.wfile.write(json.dumps(response).encode())
                    return
                
                engine = create_engine(database_url)
                SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
                
                with SessionLocal() as session:
                    query = text("""
                        SELECT id, repo_name, file_path, description, stars, 
                               owner_type, owner_login, latest_commit_date,
                               email, contact_source, contact_extracted_at
                        FROM marker_hits
                    """)
                    
                    if has_email is not None:
                        if has_email.lower() == 'true':
                            query = text("""
                                SELECT id, repo_name, file_path, description, stars, 
                                       owner_type, owner_login, latest_commit_date,
                                       email, contact_source, contact_extracted_at
                                FROM marker_hits
                                WHERE email IS NOT NULL AND email != ''
                            """)
                        else:
                            query = text("""
                                SELECT id, repo_name, file_path, description, stars, 
                                       owner_type, owner_login, latest_commit_date,
                                       email, contact_source, contact_extracted_at
                                FROM marker_hits
                                WHERE email IS NULL OR email = ''
                            """)
                    
                    if sort_by == 'stars':
                        query = text(str(query) + " ORDER BY stars DESC NULLS LAST")
                    elif sort_by == 'latest_commit_date':
                        query = text(str(query) + " ORDER BY latest_commit_date DESC NULLS LAST")
                    else:
                        query = text(str(query) + " ORDER BY id")
                    
                    query = text(str(query) + f" LIMIT {limit} OFFSET {offset}")
                    
                    result = session.execute(query)
                    rows = result.fetchall()
                    
                    hits = []
                    for row in rows:
                        hit = {
                            'id': row[0],
                            'repo_name': row[1],
                            'file_path': row[2],
                            'description': row[3] or "",
                            'stars': row[4],
                            'owner_type': row[5] or "",
                            'owner_login': row[6] or "",
                            'latest_commit_date': row[7] or "",
                            'email': row[8] or "",
                            'contact_source': row[9] or "",
                            'contact_extracted_at': row[10] or ""
                        }
                        hits.append(hit)
                    
                    response = hits
                    
            except Exception as e:
                response = {"error": f"Database error: {str(e)}"}
                self.wfile.write(json.dumps(response).encode())
                return
        else:
            response = {"message": "Endpoint not implemented", "path": path}
            
        self.wfile.write(json.dumps(response).encode())
        return

    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        if path == '/api/run-scraper':
            try:
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                
                from github_api_scraper import GitHubAPIScraper, init_database
                
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                body = json.loads(post_data.decode('utf-8'))
                
                github_token = body.get('github_token')
                extract_contacts = body.get('extract_contacts', True)
                
                init_database()
                scraper = GitHubAPIScraper(token=github_token)
                
                results = scraper.search_ai_code_generator_files_to_db(
                    max_repos_per_pattern=10,
                    min_stars=0,
                    extract_contacts=extract_contacts
                )
                
                response = {
                    "message": f"Scraper completed successfully! Added {results['total_new_records']} new records.",
                    "new_records": results['total_new_records']
                }
                
            except Exception as e:
                response = {"error": f"Scraper failed: {str(e)}"}
                self.wfile.write(json.dumps(response).encode())
                return
        else:
            response = {"error": "Endpoint not found"}
            
        self.wfile.write(json.dumps(response).encode())
        return

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        return 