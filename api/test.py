#!/usr/bin/env python3
"""
Minimal test handler for Vercel
"""
import json
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        response = {
            "status": "ok",
            "message": "Test handler working"
        }
        
        self.wfile.write(json.dumps(response).encode('utf-8'))
