#!/usr/bin/env python3
"""
Vercel serverless function handler
"""
import os
import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

def handler(request, context):
    """Vercel serverless function handler"""
    
    # Get the path from the request
    path = request.get('path', '')
    
    # Set CORS headers
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
    }
    
    if path == '/api/health':
        return handle_health()
    elif path == '/api/markers':
        return handle_markers()
    elif path == '/api/owner_types':
        return handle_owner_types()
    elif path == '/api/owner_logins':
        return handle_owner_logins()
    elif path == '/api/contact-stats':
        return handle_contact_stats()
    elif path == '/api/hits':
        return handle_hits(request.get('query', {}))
    else:
        return {
            'statusCode': 404,
            'body': json.dumps({'error': 'Not Found'}),
            'headers': headers
        }

def handle_health():
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
            return {
                'statusCode': 500,
                'body': json.dumps(response),
                'headers': {'Content-Type': 'application/json'}
            }
        
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
        return {
            'statusCode': 200,
            'body': json.dumps(response),
            'headers': {'Content-Type': 'application/json'}
        }
            
    except Exception as e:
        response = {
            "status": "error",
            "message": f"Error checking environment: {str(e)}",
            "database": "unavailable"
        }
        return {
            'statusCode': 500,
            'body': json.dumps(response),
            'headers': {'Content-Type': 'application/json'}
        }

def handle_markers():
    """Handle markers endpoint"""
    response = {
        "markers": [
            "ai-code-generator",
            "ai-code-gen",
            "code-generator",
            "ai-generator"
        ]
    }
    return {
        'statusCode': 200,
        'body': json.dumps(response),
        'headers': {'Content-Type': 'application/json'}
    }

def handle_owner_types():
    """Handle owner types endpoint"""
    response = {
        "owner_types": [
            "User",
            "Organization"
        ]
    }
    return {
        'statusCode': 200,
        'body': json.dumps(response),
        'headers': {'Content-Type': 'application/json'}
    }

def handle_owner_logins():
    """Handle owner logins endpoint"""
    response = {
        "owner_logins": [
            "example-user",
            "example-org"
        ]
    }
    return {
        'statusCode': 200,
        'body': json.dumps(response),
        'headers': {'Content-Type': 'application/json'}
    }

def handle_contact_stats():
    """Handle contact stats endpoint"""
    response = {
        "status": "not_implemented",
        "message": "Database connection required"
    }
    return {
        'statusCode': 500,
        'body': json.dumps(response),
        'headers': {'Content-Type': 'application/json'}
    }

def handle_hits(query_params):
    """Handle hits endpoint"""
    response = {
        "status": "not_implemented",
        "message": "Database connection required"
    }
    return {
        'statusCode': 500,
        'body': json.dumps(response),
        'headers': {'Content-Type': 'application/json'}
    } 