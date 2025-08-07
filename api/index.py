def handler(request, context):
    """Vercel serverless function handler"""
    import json
    import os
    from urllib.parse import urlparse, parse_qs
    
    # Set CORS headers
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
    }
    
    # Get the path from the request
    path = request.get('path', '')
    
    # Route to appropriate endpoint
    if path == '/api/health':
        try:
            # Import database modules
            from sqlalchemy import create_engine, text
            import psycopg2
            
            # Get database URL from environment
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                response = {"status": "error", "message": "Database URL not configured", "database": "unavailable"}
                return {
                    'statusCode': 500,
                    'headers': headers,
                    'body': json.dumps(response)
                }
            
            # Test database connection
            engine = create_engine(database_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM marker_hits"))
                count = result.scalar()
                response = {"status": "ok", "message": "Server is running", "database": "connected", "record_count": count}
                
        except Exception as e:
            response = {"status": "error", "message": f"Database error: {str(e)}", "database": "unavailable"}
            return {
                'statusCode': 500,
                'headers': headers,
                'body': json.dumps(response)
            }
    elif path == '/api/markers':
        response = [".claude", ".cursor", ".copilot"]
    elif path == '/api/owner_types':
        response = ["User", "Organization"]
    elif path == '/api/owner_logins':
        response = ["example-user", "example-org"]
    elif path == '/api/contact-stats':
        try:
            # Import database modules
            from sqlalchemy import create_engine, text
            from sqlalchemy.orm import sessionmaker
            import psycopg2
            
            # Get database URL from environment
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                response = {"error": "Database URL not configured"}
                return {
                    'statusCode': 500,
                    'headers': headers,
                    'body': json.dumps(response)
                }
            
            # Create engine and session
            engine = create_engine(database_url)
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            
            # Query database for stats
            with SessionLocal() as session:
                # Get total records
                total_result = session.execute(text("SELECT COUNT(*) FROM marker_hits"))
                total_records = total_result.scalar()
                
                # Get records with email
                email_result = session.execute(text("SELECT COUNT(*) FROM marker_hits WHERE email IS NOT NULL AND email != ''"))
                records_with_email = email_result.scalar()
                
                # Get records with any contact
                contact_result = session.execute(text("SELECT COUNT(*) FROM marker_hits WHERE email IS NOT NULL AND email != '' OR contact_source IS NOT NULL AND contact_source != ''"))
                records_with_any_contact = contact_result.scalar()
                
                # Calculate percentages
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
            return {
                'statusCode': 500,
                'headers': headers,
                'body': json.dumps(response)
            }
    elif path == '/api/hits':
        # Parse query parameters
        query_string = request.get('queryStringParameters', {}) or {}
        limit = int(query_string.get('limit', 10))
        offset = int(query_string.get('offset', 0))
        sort_by = query_string.get('sort_by', 'id')
        has_email = query_string.get('has_email')
        
        try:
            # Import database modules
            from sqlalchemy import create_engine, text
            from sqlalchemy.orm import sessionmaker
            import psycopg2
            
            # Get database URL from environment
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                response = {"error": "Database URL not configured"}
                return {
                    'statusCode': 500,
                    'headers': headers,
                    'body': json.dumps(response)
                }
            
            # Create engine and session
            engine = create_engine(database_url)
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            
            # Query database
            with SessionLocal() as session:
                # Build query
                query = text("""
                    SELECT id, repo_name, file_path, description, stars, 
                           owner_type, owner_login, latest_commit_date,
                           email, contact_source, contact_extracted_at
                    FROM marker_hits
                """)
                
                # Add WHERE clause for email filter
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
                
                # Add ORDER BY
                if sort_by == 'stars':
                    query = text(str(query) + " ORDER BY stars DESC NULLS LAST")
                elif sort_by == 'latest_commit_date':
                    query = text(str(query) + " ORDER BY latest_commit_date DESC NULLS LAST")
                else:
                    query = text(str(query) + " ORDER BY id")
                
                # Add LIMIT and OFFSET
                query = text(str(query) + f" LIMIT {limit} OFFSET {offset}")
                
                # Execute query
                result = session.execute(query)
                rows = result.fetchall()
                
                # Convert to list of dictionaries
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
            return {
                'statusCode': 500,
                'headers': headers,
                'body': json.dumps(response)
            }
    elif path == '/api/run-scraper':
        # Handle POST request for running the scraper
        if request.get('method', 'GET') == 'POST':
            try:
                # Import scraper modules
                import sys
                import os
                # Add parent directory to path to import github_api_scraper
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                
                from github_api_scraper import GitHubAPIScraper, init_database
                
                # Get request body
                body = request.get('body', '{}')
                if isinstance(body, str):
                    import json
                    body = json.loads(body)
                
                # Extract parameters
                github_token = body.get('github_token')
                extract_contacts = body.get('extract_contacts', True)
                
                # Initialize database
                init_database()
                
                # Create scraper instance
                scraper = GitHubAPIScraper(token=github_token)
                
                # Run the scraper
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
                return {
                    'statusCode': 500,
                    'headers': headers,
                    'body': json.dumps(response)
                }
        else:
            response = {"error": "Method not allowed"}
            return {
                'statusCode': 405,
                'headers': headers,
                'body': json.dumps(response)
            }
    else:
        response = {"message": "Endpoint not implemented", "path": path}
    
    return {
        'statusCode': 200,
        'headers': headers,
        'body': json.dumps(response)
    } 