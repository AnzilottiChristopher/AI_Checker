#!/usr/bin/env python3
"""
Script to fix database schema issues by ensuring all datetime fields are stored as strings.
"""

import sqlite3
from datetime import datetime

def fix_database_schema():
    """Fix database schema by converting datetime objects to strings."""
    conn = sqlite3.connect('ai_code_generator.db')
    cursor = conn.cursor()
    
    print("Checking database for datetime objects...")
    
    # Check for records with datetime objects
    cursor.execute('SELECT id, contact_extracted_at, latest_commit_date FROM marker_hits')
    results = cursor.fetchall()
    
    fixed_count = 0
    for row in results:
        record_id, contact_extracted_at, latest_commit_date = row
        
        needs_update = False
        new_contact_extracted_at = contact_extracted_at
        new_latest_commit_date = latest_commit_date
        
        # Check if contact_extracted_at is a datetime object
        if contact_extracted_at is not None and not isinstance(contact_extracted_at, str):
            try:
                new_contact_extracted_at = contact_extracted_at.isoformat()
                needs_update = True
                print(f"Converting contact_extracted_at for ID {record_id}: {contact_extracted_at} -> {new_contact_extracted_at}")
            except:
                pass
        
        # Check if latest_commit_date is a datetime object
        if latest_commit_date is not None and not isinstance(latest_commit_date, str):
            try:
                new_latest_commit_date = latest_commit_date.isoformat()
                needs_update = True
                print(f"Converting latest_commit_date for ID {record_id}: {latest_commit_date} -> {new_latest_commit_date}")
            except:
                pass
        
        # Update the record if needed
        if needs_update:
            cursor.execute('''
                UPDATE marker_hits 
                SET contact_extracted_at = ?, latest_commit_date = ?
                WHERE id = ?
            ''', (new_contact_extracted_at, new_latest_commit_date, record_id))
            fixed_count += 1
    
    conn.commit()
    conn.close()
    
    print(f"Fixed {fixed_count} records with datetime objects.")

if __name__ == "__main__":
    fix_database_schema() 