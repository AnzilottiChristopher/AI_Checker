#!/usr/bin/env python3
"""
Script to fix NULL description values in the database.
"""

import sqlite3

def fix_null_descriptions():
    """Replace NULL description values with empty strings."""
    conn = sqlite3.connect('ai_code_generator.db')
    cursor = conn.cursor()
    
    print("Checking for NULL description values...")
    
    # Count records with NULL descriptions
    cursor.execute('SELECT COUNT(*) FROM marker_hits WHERE description IS NULL')
    null_count = cursor.fetchone()[0]
    print(f"Found {null_count} records with NULL descriptions")
    
    if null_count > 0:
        print("Replacing NULL descriptions with empty strings...")
        cursor.execute('UPDATE marker_hits SET description = "" WHERE description IS NULL')
        conn.commit()
        print(f"Updated {null_count} records")
    else:
        print("No NULL descriptions found")
    
    # Verify the fix
    cursor.execute('SELECT COUNT(*) FROM marker_hits WHERE description IS NULL')
    remaining_null = cursor.fetchone()[0]
    print(f"Remaining NULL descriptions: {remaining_null}")
    
    conn.close()
    
    print("Database fix completed!")

if __name__ == "__main__":
    fix_null_descriptions() 