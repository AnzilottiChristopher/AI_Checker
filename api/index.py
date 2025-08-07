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
            response = {"status": "ok", "message": "Server is running", "database": "unavailable"}
        elif path == '/api/markers':
            response = [".claude", ".cursor", ".copilot"]
        elif path == '/api/owner_types':
            response = ["User", "Organization"]
        elif path == '/api/owner_logins':
            response = ["example-user", "example-org"]
        elif path == '/api/contact-stats':
            response = {
                "total_records": 0, "records_with_email": 0, "records_with_any_contact": 0,
                "email_percentage": 0, "any_contact_percentage": 0
            }
        else:
            response = {"message": "Endpoint not implemented", "path": path}
            
        self.wfile.write(json.dumps(response).encode())
        return

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        return 