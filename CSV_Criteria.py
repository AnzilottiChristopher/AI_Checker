import csv
import re
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import os
import sys
from bs4 import BeautifulSoup

class CSVCriteriaOrganizer:
    """
    Organizes CSV data based on scoring criteria for repository evaluation.
    """
    
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.data = []
        self.scores = []
        
        # Free email providers (not professional)
        self.free_email_providers = {
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com',
            'icloud.com', 'protonmail.com', 'tutanota.com', 'mail.com',
            'yandex.com', 'zoho.com', 'gmx.com', 'live.com', 'msn.com'
        }
        
        # AI/Artificial Intelligence keywords
        self.ai_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning',
            'neural network', 'nlp', 'natural language processing', 'computer vision',
            'data science', 'predictive analytics', 'automation', 'chatbot',
            'recommendation system', 'pattern recognition', 'algorithm'
        ]
        
        # Product/Platform keywords
        self.product_keywords = [
            'product', 'platform', 'saas', 'software as a service', 'app',
            'application', 'tool', 'service', 'solution', 'marketplace',
            'dashboard', 'api', 'sdk', 'library', 'framework', 'plugin',
            'extension', 'widget', 'component', 'module', 'package'
        ]
        
        # Website link patterns
        self.website_patterns = [
            r'https?://[^\s]+',
            r'www\.[^\s]+',
            r'[^\s]+\.(com|org|net|io|co|ai|dev|app|tech|digital|online)'
        ]

    def load_csv_data(self) -> bool:
        """
        Load CSV data from file.
        """
        try:
            with open(self.csv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                self.data = list(reader)
            
            if len(self.data) < 2:
                print("Error: CSV file must have at least a header row and one data row.")
                return False
                
            print(f"Loaded {len(self.data)} rows from {self.csv_file_path}")
            return True
            
        except FileNotFoundError:
            print(f"Error: File {self.csv_file_path} not found.")
            return False
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return False

    def is_professional_email(self, email: str) -> bool:
        """
        Check if email is from a professional domain (not free provider).
        """
        if not email or '@' not in email:
            return False
            
        domain = email.split('@')[-1].lower()
        # Professional emails are those NOT from free providers
        return domain not in self.free_email_providers

    def has_website_link(self, text: str) -> bool:
        """
        Check if text contains website links.
        """
        if not text:
            return False
            
        text_lower = text.lower()
        for pattern in self.website_patterns:
            if re.search(pattern, text_lower):
                return True
        return False

    def has_ai_keywords(self, text: str) -> bool:
        """
        Check if text contains AI/Artificial Intelligence keywords.
        """
        if not text:
            return False
            
        text_lower = text.lower()
        for keyword in self.ai_keywords:
            if keyword in text_lower:
                return True
        return False

    def has_product_keywords(self, text: str) -> bool:
        """
        Check if text contains product/platform keywords.
        """
        if not text:
            return False
            
        text_lower = text.lower()
        for keyword in self.product_keywords:
            if keyword in text_lower:
                return True
        return False

    def is_active_last_90_days(self, last_activity: str) -> bool:
        """
        Check if repository was active in the last 90 days.
        """
        if not last_activity:
            return False
            
        try:
            # Try to parse the date
            if 'T' in last_activity:
                # ISO format: 2023-12-01T10:30:00Z
                activity_date = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
            else:
                # Try other common formats
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                    try:
                        activity_date = datetime.strptime(last_activity, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return False
            
            ninety_days_ago = datetime.now() - timedelta(days=90)
            return activity_date > ninety_days_ago
            
        except Exception:
            return False

    def fetch_repo_data(self, repo_url: str) -> Dict:
        """
        Fetch repository data by parsing GitHub HTML page.
        """
        if not repo_url or 'github.com' not in repo_url:
            return {}
            
        try:
            # Add headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(repo_url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"Warning: Could not fetch {repo_url}, status code: {response.status_code}")
                return {}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract repository data
            repo_data = {}
            
            # Get repository description
            description_elem = soup.find('p', {'class': 'f4'})
            if description_elem:
                repo_data['description'] = description_elem.get_text(strip=True)
            
            # Get README content
            readme_elem = soup.find('div', {'id': 'readme'})
            if readme_elem:
                readme_content = readme_elem.get_text(strip=True)
                repo_data['readme_content'] = readme_content
            
            # Get last commit date
            commit_elem = soup.find('relative-time')
            if commit_elem:
                repo_data['last_commit'] = commit_elem.get('datetime')
            
            # Get repository topics/tags
            topics = []
            topic_elems = soup.find_all('a', {'class': 'topic-tag'})
            for topic in topic_elems:
                topics.append(topic.get_text(strip=True))
            repo_data['topics'] = topics
            
            # Get website link from description or README
            website_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('http') and 'github.com' not in href:
                    website_links.append(href)
            repo_data['website_links'] = website_links
            
            return repo_data
                    
        except Exception as e:
            print(f"Warning: Could not fetch repo data for {repo_url}: {e}")
            
        return {}

    def calculate_score(self, row: List[str]) -> int:
        """
        Calculate score for a row based on the specified criteria.
        """
        score = 0
        
        # Handle both 18-column and 9-column structures
        if len(row) < 9:
            print(f"Warning: Row has only {len(row)} columns, minimum 9 required")
            return score
        
        # Professional email (column H - actual email column)
        email = row[7] if len(row) > 7 else ""  # Email is in column H (index 7)
        if self.is_professional_email(email):
            score += 2
            print(f"  +2: Professional email ({email})")
        elif email and '@' in email:
            print(f"  +0: Free email provider ({email})")
        
        # Repository description from CSV column I
        description = row[8] if len(row) > 8 else ""  # Description is in column I (index 8)
        readme_content = ""
        
        # Look for README content in other columns
        for i, cell in enumerate(row):
            if cell and len(cell) > 10:  # Non-empty cell with substantial content
                if 'readme' in str(i).lower() or 'content' in str(i).lower():
                    readme_content = cell
        
        # Try to fetch additional data from GitHub if repository URL is available
        repo_url = row[1] if len(row) > 1 else ""  # Repository URL is in column B (index 1)
        repo_name = ""
        
        # Extract repo name from URL if needed
        if repo_url and 'github.com' in repo_url:
            # Extract owner/repo from URL like https://github.com/owner/repo
            parts = repo_url.rstrip('/').split('/')
            if len(parts) >= 2:
                repo_name = f"{parts[-2]}/{parts[-1]}"
            print(f"  Fetching data from: {repo_url}")
            repo_data = self.fetch_repo_data(repo_url)
        else:
            # Fallback: try to construct URL from repo name if URL is not provided
            repo_name = row[1] if len(row) > 1 else ""
            if repo_name and '/' in repo_name:
                repo_url = f"https://github.com/{repo_name}"
                print(f"  Fetching data from: {repo_url}")
                repo_data = self.fetch_repo_data(repo_url)
            else:
                repo_data = {}
        
        # Use HTML-parsed data if available, otherwise fall back to CSV data
        if repo_data.get('description'):
            description = repo_data['description']
            print(f"  Found description from GitHub: {description[:100]}...")
        
        if repo_data.get('readme_content'):
            readme_content = repo_data['readme_content']
            print(f"  Found README content from GitHub: {len(readme_content)} characters")
        
        # Check for website links in parsed data
        if repo_data.get('website_links'):
            score += 1
            print(f"  +1: Website link found in repository ({len(repo_data['website_links'])} links)")
        
        # Check last commit date from parsed data
        if repo_data.get('last_commit'):
            if self.is_active_last_90_days(repo_data['last_commit']):
                score += 1
                print(f"  +1: Active in last 90 days (last commit: {repo_data['last_commit']})")
        
        # Website link in description (if not already found from HTML)
        if not repo_url or not repo_data.get('website_links'):
            if self.has_website_link(description):
                score += 1
                print(f"  +1: Website link in description")
        
        # AI keywords in README
        if self.has_ai_keywords(readme_content):
            score += 3
            print(f"  +3: AI keywords in README")
        
        # Product keywords in description or README
        combined_text = f"{description} {readme_content}"
        if self.has_product_keywords(combined_text):
            score += 2
            print(f"  +2: Product/Platform keywords found")
        
        # Debug information
        if description:
            print(f"  Description: {description[:100]}...")
        if readme_content:
            print(f"  README content: {len(readme_content)} characters")
        
        # Active last 90 days (from CSV column E - Latest Commit)
        if not repo_url or not repo_data.get('last_commit'):
            last_activity = row[4] if len(row) > 4 else ""  # Latest Commit is in column E (index 4)
            
            if self.is_active_last_90_days(last_activity):
                score += 1
                print(f"  +1: Active in last 90 days (from CSV data: {last_activity})")
        
        return score

    def organize_csv(self, output_file: str = None) -> bool:
        """
        Organize CSV data based on scoring criteria.
        """
        if not self.load_csv_data():
            return False
        
        if output_file is None:
            base_name = os.path.splitext(self.csv_file_path)[0]
            output_file = f"{base_name}_organized.csv"
        
        print("Calculating scores for each row...")
        
        # Calculate scores for all rows (skip header)
        header = self.data[0]
        data_rows = self.data[1:]
        
        # Always use column 17 (index 17) for score, regardless of structure
        # Extend header to 18 columns if needed
        while len(header) < 18:
            header.append("")
        header[17] = "Score"
        
        scored_rows = []
        for i, row in enumerate(data_rows, start=2):  # Start at 2 since we skipped header
            print(f"\nRow {i}:")
            score = self.calculate_score(row)
            
            # Always use column 17 (index 17) for score, regardless of structure
            # Extend row to 18 columns if needed
            while len(row) < 18:
                row.append("")
            row[17] = str(score)
            
            scored_rows.append((score, row))
            print(f"  Total score: {score}")
        
        # Sort by score (highest first)
        scored_rows.sort(key=lambda x: x[0], reverse=True)
        
        # Write organized CSV
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                
                # Write header
                writer.writerow(header)
                
                # Write sorted data
                for score, row in scored_rows:
                    writer.writerow(row)
            
            print(f"\nOrganized CSV saved to: {output_file}")
            print(f"Total rows processed: {len(scored_rows)}")
            
            # Show top 10 results
            print("\nTop 10 highest scoring entries:")
            for i, (score, row) in enumerate(scored_rows[:10], 1):
                identifier = row[0] if row else "Unknown"
                print(f"{i}. Score {score}: {identifier}")
            
            return True
            
        except Exception as e:
            print(f"Error writing organized CSV: {e}")
            return False

    def analyze_csv_structure(self):
        """
        Analyze the structure of the CSV file to help identify columns.
        """
        if not self.load_csv_data():
            return
        
        print("CSV Structure Analysis:")
        print("=" * 50)
        
        header = self.data[0]
        print(f"Header row ({len(header)} columns):")
        for i, col in enumerate(header):
            print(f"  Column {chr(65+i)} ({i}): {col}")
        
        if len(self.data) > 1:
            sample_row = self.data[1]
            print(f"\nSample data row ({len(sample_row)} columns):")
            for i, cell in enumerate(sample_row):
                preview = cell[:50] + "..." if len(cell) > 50 else cell
                print(f"  Column {chr(65+i)} ({i}): {preview}")

def main():
    """
    Main function to run the CSV organization program.
    """
    if len(sys.argv) < 2:
        print("Usage: python CSV_Criteria.py <csv_file_path> [output_file_path]")
        print("Example: python CSV_Criteria.py data.csv organized_data.csv")
        return
    
    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    organizer = CSVCriteriaOrganizer(csv_file)
    
    # First, analyze the structure
    organizer.analyze_csv_structure()
    
    # Then organize the data
    print("\n" + "=" * 50)
    print("Starting CSV organization...")
    print("=" * 50)
    
    success = organizer.organize_csv(output_file)
    
    if success:
        print("\nCSV organization completed successfully!")
    else:
        print("\nCSV organization failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
