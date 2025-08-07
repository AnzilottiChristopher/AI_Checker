#!/usr/bin/env python3
"""
Script to add missing columns to the database schema.
"""

import sqlite3

def add_missing_columns():
    """Add missing columns to the database."""
    conn = sqlite3.connect('ai_code_generator.db')
    cursor = conn.cursor()
    
    print("Checking for missing columns...")
    
    # Get current columns
    cursor.execute('PRAGMA table_info(marker_hits)')
    existing_columns = [row[1] for row in cursor.fetchall()]
    print(f"Existing columns: {existing_columns}")
    
    # Add missing columns
    if 'owner_type' not in existing_columns:
        print("Adding owner_type column...")
        cursor.execute('ALTER TABLE marker_hits ADD COLUMN owner_type TEXT')
    
    if 'owner_login' not in existing_columns:
        print("Adding owner_login column...")
        cursor.execute('ALTER TABLE marker_hits ADD COLUMN owner_login TEXT')
    
    if 'owner_email' not in existing_columns:
        print("Adding owner_email column...")
        cursor.execute('ALTER TABLE marker_hits ADD COLUMN owner_email TEXT')
    
    # Verify the changes
    cursor.execute('PRAGMA table_info(marker_hits)')
    updated_columns = [row[1] for row in cursor.fetchall()]
    print(f"Updated columns: {updated_columns}")
    
    conn.commit()
    conn.close()
    
    print("Database schema updated successfully!")

if __name__ == "__main__":
    add_missing_columns() 