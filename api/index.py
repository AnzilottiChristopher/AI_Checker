#!/usr/bin/env python3
"""
Vercel serverless function handler
"""
import os
import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

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
        """Handle health check with environment variable test"""
        try:
            # Check if DATABASE_URL exists
            database_url = os.getenv('DATABASE_URL')
            
            if not database_url:
                response = {
                    "status": "error",
                    "message": "DATABASE_URL environment variable is not set",
                    "database": "unavailable",
                    "debug_info": {
                        "database_url_exists": False,
                        "available_env_vars": list(os.environ.keys())
                    }
                }
                self.send_json_response(response, 500)
                return
            
            # If we get here, DATABASE_URL exists
            response = {
                "status": "partial_success",
                "message": "DATABASE_URL is set, but database connection not tested yet",
                "database": "url_found",
                "debug_info": {
                    "database_url_exists": True,
                    "database_url_length": len(database_url),
                    "database_url_start": database_url[:20] + "..." if len(database_url) > 20 else database_url
                }
            }
            self.send_json_response(response, 200)
                
        except Exception as e:
            response = {
                "status": "error",
                "message": f"Error checking environment: {str(e)}",
                "database": "unavailable"
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
        """Handle contact stats endpoint"""
        response = {
            "status": "not_implemented",
            "message": "Database connection required"
        }
        self.send_json_response(response, 500)
    
    def handle_hits(self, query_params):
        """Handle hits endpoint"""
        response = {
            "status": "not_implemented",
            "message": "Database connection required"
        }
        self.send_json_response(response, 500)
    
    def handle_run_scraper(self):
        """Handle run scraper endpoint"""
        response = {
            "status": "not_implemented",
            "message": "Database connection required"
        }
        self.send_json_response(response, 500)
    
    def send_json_response(self, data, status_code):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8')) 