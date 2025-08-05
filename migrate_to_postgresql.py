#!/usr/bin/env python3
"""
Migration script to transfer data from SQLite to PostgreSQL.
This script helps migrate existing data when switching from SQLite to PostgreSQL.
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def migrate_sqlite_to_postgresql():
    """
    Migrate data from SQLite to PostgreSQL.
    Requires DATABASE_URL environment variable to be set for PostgreSQL.
    """
    
    # Check if PostgreSQL DATABASE_URL is set
    postgresql_url = os.getenv("DATABASE_URL")
    if not postgresql_url or not postgresql_url.startswith("postgresql"):
        logger.error("DATABASE_URL environment variable not set or not a PostgreSQL URL")
        logger.info("Please set DATABASE_URL to your PostgreSQL connection string")
        logger.info("Example: postgresql://user:password@host:port/database")
        return False
    
    # SQLite database path
    sqlite_path = "./ai_code_generator.db"
    if not os.path.exists(sqlite_path):
        logger.error(f"SQLite database not found at {sqlite_path}")
        return False
    
    try:
        # Connect to SQLite
        logger.info("Connecting to SQLite database...")
        sqlite_engine = create_engine(f"sqlite:///{sqlite_path}")
        sqlite_session = sessionmaker(bind=sqlite_engine)()
        
        # Connect to PostgreSQL
        logger.info("Connecting to PostgreSQL database...")
        postgresql_engine = create_engine(postgresql_url, pool_pre_ping=True, pool_recycle=300)
        postgresql_session = sessionmaker(bind=postgresql_engine)()
        
        # Check if tables exist in SQLite
        result = sqlite_session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='marker_hits'"))
        if not result.fetchone():
            logger.error("No marker_hits table found in SQLite database")
            return False
        
        # Get data from SQLite
        logger.info("Reading data from SQLite...")
        result = sqlite_session.execute(text("SELECT * FROM marker_hits"))
        rows = result.fetchall()
        
        if not rows:
            logger.warning("No data found in SQLite database")
            return True
        
        logger.info(f"Found {len(rows)} records to migrate")
        
        # Create table in PostgreSQL if it doesn't exist
        logger.info("Creating table in PostgreSQL...")
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS marker_hits (
            id SERIAL PRIMARY KEY,
            marker VARCHAR,
            repo_name VARCHAR,
            repo_url VARCHAR,
            file_path VARCHAR,
            file_url VARCHAR,
            stars INTEGER,
            description VARCHAR,
            owner_type VARCHAR,
            owner_login VARCHAR,
            owner_email VARCHAR,
        
            contact_source VARCHAR,
            contact_extracted_at TIMESTAMP
        );
        """
        postgresql_session.execute(text(create_table_sql))
        postgresql_session.commit()
        
        # Insert data into PostgreSQL
        logger.info("Inserting data into PostgreSQL...")
        insert_sql = """
        INSERT INTO marker_hits 
        (marker, repo_name, repo_url, file_path, file_url, stars, description, 
         owner_type, owner_login, owner_email, contact_source, contact_extracted_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        for row in rows:
            # Convert SQLite row to tuple for PostgreSQL insert
            data = (
                row[1],  # marker
                row[2],  # repo_name
                row[3],  # repo_url
                row[4],  # file_path
                row[5],  # file_url
                row[6],  # stars
                row[7],  # description
                row[8],  # owner_type
                row[9],  # owner_login
                row[10], # owner_email
    
                row[12], # contact_source
                row[13]  # contact_extracted_at
            )
            postgresql_session.execute(text(insert_sql), data)
        
        postgresql_session.commit()
        logger.info(f"Successfully migrated {len(rows)} records to PostgreSQL")
        
        # Verify migration
        result = postgresql_session.execute(text("SELECT COUNT(*) FROM marker_hits"))
        count = result.fetchone()[0]
        logger.info(f"PostgreSQL now contains {count} records")
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        return False
    finally:
        if 'sqlite_session' in locals():
            sqlite_session.close()
        if 'postgresql_session' in locals():
            postgresql_session.close()

def main():
    """Main function to run the migration."""
    logger.info("Starting SQLite to PostgreSQL migration...")
    
    success = migrate_sqlite_to_postgresql()
    
    if success:
        logger.info("Migration completed successfully!")
        logger.info("You can now use PostgreSQL as your database")
        logger.info("Remember to set DATABASE_URL environment variable for your application")
    else:
        logger.error("Migration failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 