#!/usr/bin/env python3
"""
Script to populate top contributor data for all existing repositories in the database.
This script will:
1. Query all repositories from the database
2. For each repository, fetch the top contributor and their email
3. Update the database with this information
"""

import os
import sys
import time
import requests
from typing import Optional, Tuple
from github_api_scraper import SessionLocal, MarkerHit, GitHubAPIScraper
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TopContributorPopulator:
    def __init__(self, github_token=None, backup_tokens=None):
        if github_token:
            self.scraper = GitHubAPIScraper(github_token, backup_tokens or [])
        else:
            # Try to get token from environment variable
            import os
            env_token = os.getenv('GITHUB_TOKEN')
            if env_token:
                self.scraper = GitHubAPIScraper(env_token)
            else:
                raise ValueError("GitHub token is required. Either pass it to the constructor or set GITHUB_TOKEN environment variable.")
        self.session = SessionLocal()
        
    def get_top_contributor(self, repo_name: str) -> Optional[Tuple[str, str]]:
        """
        Get the top contributor and their email for a repository.
        Returns (username, email) or None if not found.
        """
        try:
            # Extract owner and repo from full name (e.g., "owner/repo")
            if '/' not in repo_name:
                logger.warning(f"Invalid repo name format: {repo_name}")
                return None
                
            owner, repo = repo_name.split('/', 1)
            
            # Get contributors for the repository
            contributors_url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
            headers = self.scraper.get_headers()
            
            response = requests.get(contributors_url, headers=headers)
            
            if response.status_code == 404:
                logger.warning(f"Repository not found: {repo_name}")
                return None
            elif response.status_code != 200:
                logger.warning(f"Failed to get contributors for {repo_name}: {response.status_code}")
                return None
                
            contributors = response.json()
            
            if not contributors:
                logger.info(f"No contributors found for {repo_name}")
                return None
                
            # Get the top contributor (first in the list)
            top_contributor = contributors[0]
            username = top_contributor.get('login')
            
            if not username:
                logger.warning(f"No username found for top contributor in {repo_name}")
                return None
                
            # Get the user's profile to extract email
            user_url = f"https://api.github.com/users/{username}"
            user_response = requests.get(user_url, headers=headers)
            
            if user_response.status_code == 200:
                user_data = user_response.json()
                email = user_data.get('email')
                
                if email:
                    logger.info(f"Found top contributor for {repo_name}: {username} ({email})")
                    return username, email
                else:
                    logger.info(f"Found top contributor for {repo_name}: {username} (no email)")
                    return username, None
            else:
                logger.warning(f"Failed to get user profile for {username}: {user_response.status_code}")
                return username, None
                
        except Exception as e:
            logger.error(f"Error getting top contributor for {repo_name}: {e}")
            return None
    
    def populate_top_contributors(self, limit: Optional[int] = None):
        """
        Populate top contributor data for all repositories in the database.
        
        Args:
            limit: Optional limit on number of repositories to process
        """
        try:
            # Query repositories that don't have top contributor data yet
            query = self.session.query(MarkerHit).filter(
                MarkerHit.top_contributor.is_(None)
            ).distinct(MarkerHit.repo_name)
            
            if limit:
                query = query.limit(limit)
                
            repositories = query.all()
            
            logger.info(f"Found {len(repositories)} repositories to process")
            
            if not repositories:
                logger.info("No repositories found that need top contributor data")
                return
                
            processed = 0
            updated = 0
            
            for hit in repositories:
                try:
                    logger.info(f"Processing {processed + 1}/{len(repositories)}: {hit.repo_name}")
                    
                    # Get top contributor data
                    contributor_data = self.get_top_contributor(hit.repo_name)
                    
                    if contributor_data:
                        username, email = contributor_data
                        
                        # Update all records for this repository
                        update_count = self.session.query(MarkerHit).filter(
                            MarkerHit.repo_name == hit.repo_name
                        ).update({
                            'top_contributor': username,
                            'top_contributor_email': email
                        })
                        
                        self.session.commit()
                        updated += update_count
                        logger.info(f"Updated {update_count} records for {hit.repo_name}")
                    else:
                        # Mark as processed even if no contributor found
                        update_count = self.session.query(MarkerHit).filter(
                            MarkerHit.repo_name == hit.repo_name
                        ).update({
                            'top_contributor': 'none',
                            'top_contributor_email': None
                        })
                        
                        self.session.commit()
                        logger.info(f"Marked {update_count} records for {hit.repo_name} as having no contributors")
                    
                    processed += 1
                    
                    # Rate limiting - be conservative
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing {hit.repo_name}: {e}")
                    self.session.rollback()
                    continue
                    
            logger.info(f"Completed! Processed {processed} repositories, updated {updated} records")
            
        except Exception as e:
            logger.error(f"Error in populate_top_contributors: {e}")
            self.session.rollback()
        finally:
            self.session.close()

def main():
    """Main function to run the top contributor population script."""
    print("🚀 Starting Top Contributor Population Script")
    print("=" * 50)
    
    # Check if limit is provided as command line argument
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
            print(f"Processing limit: {limit} repositories")
        except ValueError:
            print("Invalid limit provided. Usage: python populate_top_contributors.py [limit]")
            return
    
    populator = TopContributorPopulator()
    populator.populate_top_contributors(limit)
    
    print("=" * 50)
    print("✅ Top Contributor Population Complete!")

if __name__ == "__main__":
    main()
