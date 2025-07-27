#!/usr/bin/env python3
"""
Usage examples for the API-only GitHub Code Analyzer
This script demonstrates how to use the GitHubAPICodeAnalyzer class for various types of repository analysis
using only the GitHub API (no local cloning required).

Each function below is a self-contained example. Uncomment the desired function call at the bottom to run it.
"""

import os
from github_api_scraper import GitHubAPICodeAnalyzer

# =========================
# Basic Usage Example
# =========================
def basic_usage():
    """Basic usage example - analyzes a few Python web scraping repos using the API only."""
    print("=== Basic API-Only Analysis ===")
    
    # Set your GitHub token for better rate limits (recommended)
    github_token = os.getenv('GITHUB_TOKEN')  # Set this in your environment
    
    # Initialize the analyzer with your token and output directory
    analyzer = GitHubAPICodeAnalyzer(github_token, output_dir="basic_analysis")
    
    # Analyze repositories matching the search query
    analyzer.analyze_repositories(
        search_query="python web scraping",   # GitHub search query
        max_repos=3,                    # Number of repos to analyze
        max_files_per_repo=25,          # Files per repo to analyze
        min_stars=5,                    # Only repos with 5+ stars
        file_size_limit_mb=0.5          # Skip files larger than 500KB
    )
    
    print("Analysis complete! Check the 'basic_analysis' directory for results.")

# =========================
# Advanced Search Examples
# =========================
def advanced_search_examples():
    """Examples of different search strategies for various domains."""
    github_token = os.getenv('GITHUB_TOKEN')
    analyzer = GitHubAPICodeAnalyzer(github_token, output_dir="advanced_analysis")
    
    # Example 1: Security-focused analysis
    print("=== Security Tools Analysis ===")
    analyzer.analyze_repositories(
        search_query="security vulnerability scanner python",
        max_repos=2,
        max_files_per_repo=40,
        min_stars=50  # Popular security tools
    )
    
    # Example 2: Machine learning repositories
    print("=== ML/AI Tools Analysis ===")
    analyzer.analyze_repositories(
        search_query="machine learning tensorflow pytorch",
        max_repos=2,
        max_files_per_repo=30,
        min_stars=100  # Well-established ML repos
    )
    
    # Example 3: Web frameworks
    print("=== Web Framework Analysis ===")
    analyzer.analyze_repositories(
        search_query="web framework flask django fastapi",
        max_repos=2,
        max_files_per_repo=35,
        min_stars=20
    )

# =========================
# Analyze Specific Repositories
# =========================
def analyze_specific_repositories():
    """Analyze specific repositories by their GitHub URLs."""
    github_token = os.getenv('GITHUB_TOKEN')
    analyzer = GitHubAPICodeAnalyzer(github_token, output_dir="specific_repos")
    
    # List of specific repositories to analyze
    repositories = [
        "https://github.com/psf/requests",
        "https://github.com/pallets/flask",
        "https://github.com/django/django"
    ]
    
    for repo_url in repositories:
        print(f"=== Analyzing {repo_url} ===")
        analyzer.analyze_specific_repository(
            repo_url=repo_url,
            max_files=50  # Analyze up to 50 files
        )
        print(f"Completed analysis of {repo_url}")

# =========================
# Language-Specific Analysis
# =========================
def language_specific_analysis():
    """Focus analysis on specific programming languages (Python, JavaScript)."""
    github_token = os.getenv('GITHUB_TOKEN')
    analyzer = GitHubAPICodeAnalyzer(github_token, output_dir="language_analysis")
    
    # Python-specific analysis
    print("=== Python Security Analysis ===")
    analyzer.analyze_repositories(
        search_query="language:python security",
        max_repos=3,
        max_files_per_repo=30,
        min_stars=10
    )
    
    # JavaScript/Node.js analysis
    print("=== JavaScript Security Analysis ===")
    analyzer.analyze_repositories(
        search_query="language:javascript security node",
        max_repos=3,
        max_files_per_repo=30,
        min_stars=10
    )

# =========================
# Batch Analysis with Different Settings
# =========================
def batch_analysis_with_different_settings():
    """Run multiple analyses with different search queries and settings."""
    github_token = os.getenv('GITHUB_TOKEN')
    
    # Define different analysis configurations
    analysis_configs = [
        {
            'name': 'crypto_tools',
            'query': 'cryptocurrency bitcoin blockchain',
            'max_repos': 3,
            'max_files': 20,
            'min_stars': 25
        },
        {
            'name': 'automation_tools',
            'query': 'python automation selenium',
            'max_repos': 4,
            'max_files': 35,
            'min_stars': 15
        },
        {
            'name': 'network_tools',
            'query': 'network scanner port python',
            'max_repos': 2,
            'max_files': 25,
            'min_stars': 10
        }
    ]
    
    # Run each analysis configuration
    for config in analysis_configs:
        print(f"=== Running {config['name']} analysis ===")
        
        analyzer = GitHubAPICodeAnalyzer(
            github_token, 
            output_dir=f"batch_{config['name']}"
        )
        
        analyzer.analyze_repositories(
            search_query=config['query'],
            max_repos=config['max_repos'],
            max_files_per_repo=config['max_files'],
            min_stars=config['min_stars'],
            file_size_limit_mb=1.0
        )
        
        print(f"Completed {config['name']} analysis")

# =========================
# Custom Suspicious Patterns Example
# =========================
def custom_suspicious_patterns():
    """Example of customizing suspicious patterns for domain-specific analysis."""
    from github_api_scraper import GitHubAPICodeAnalyzer, APICodeAnalyzer
    
    # Create a custom analyzer class with additional suspicious patterns
    class CustomAPIAnalyzer(APICodeAnalyzer):
        def __init__(self):
            super().__init__()
            # Add domain-specific suspicious patterns
            self.SUSPICIOUS_PATTERNS.extend([
                r'bitcoin|btc|ethereum|crypto',  # Cryptocurrency
                r'mining|miner|hashrate',        # Mining operations
                r'wallet|private.*key',          # Wallet operations
                r'tor|proxy|vpn|anonymiz',       # Anonymization
                r'ddos|flood|spam',              # Malicious network activity
                r'keylog|screen.*shot|steal',    # Data theft
                r'inject|exploit|payload',       # Code injection
                r'obfuscat|deobfuscat',         # Code obfuscation
            ])
    
    github_token = os.getenv('GITHUB_TOKEN')
    
    # Create analyzer with custom patterns
    analyzer = GitHubAPICodeAnalyzer(github_token, output_dir="custom_patterns")
    analyzer.analyzer = CustomAPIAnalyzer()  # Use custom analyzer
    
    print("=== Custom Pattern Analysis ===")
    analyzer.analyze_repositories(
        search_query="python hacking tools",
        max_repos=2,
        max_files_per_repo=20,
        min_stars=5
    )

# =========================
# Rate Limit Aware Analysis Example
# =========================
def rate_limit_aware_analysis():
    """Example showing how to handle GitHub API rate limits gracefully."""
    github_token = os.getenv('GITHUB_TOKEN')
    
    if not github_token:
        print("WARNING: No GitHub token provided!")
        print("You'll be limited to 60 requests/hour instead of 5000/hour")
        print("Set GITHUB_TOKEN environment variable for better performance")
        return
    
    analyzer = GitHubAPICodeAnalyzer(github_token, output_dir="rate_limit_demo")
    
    # Adjust rate limiting (increase delay between requests)
    analyzer.scraper.rate_limit_delay = 0.5  # Wait 0.5 seconds between requests
    
    print("=== Rate Limit Aware Analysis ===")
    analyzer.analyze_repositories(
        search_query="python data science",
        max_repos=5,  # More repos but with rate limiting
        max_files_per_repo=20,
        min_stars=20
    )

# =========================
# Export and Visualization Example
# =========================
def export_and_visualization_example():
    """Example showing how to export results for visualization and reporting."""
    github_token = os.getenv('GITHUB_TOKEN')
    analyzer = GitHubAPICodeAnalyzer(github_token, output_dir="export_demo")
    
    # Run analysis
    analyzer.analyze_repositories(
        search_query="python security scanner",
        max_repos=2,
        max_files_per_repo=15,
        min_stars=10
    )
    
    # The results are automatically exported to:
    # - api_analysis_results.json (structured data)
    # - api_analysis_results.csv (spreadsheet format)
    # - api_analysis_summary.txt (human-readable report)
    
    print("=== Export Complete ===")
    print("Files created:")
    print("- api_analysis_results.json (for dashboard)")
    print("- api_analysis_results.csv (for spreadsheets)")
    print("- api_analysis_summary.txt (readable report)")
    print("\nUpload the JSON file to the dashboard.html for visualization!")

# =========================
# Quick Repository Security Check
# =========================
def quick_repo_check():
    """Quick security check of a single repository (interactive)."""
    github_token = os.getenv('GITHUB_TOKEN')
    analyzer = GitHubAPICodeAnalyzer(github_token, output_dir="quick_check")
    
    # Prompt user for a repository URL
    repo_url = input("Enter GitHub repository URL: ").strip()
    
    if not repo_url:
        repo_url = "https://github.com/psf/requests"  # Default example
    
    print(f"=== Quick Security Check: {repo_url} ===")
    analyzer.analyze_specific_repository(repo_url, max_files=30)
    
    # Print immediate results
    analyzer.print_analysis_summary()

# =========================
# AI Code Generator Marker Search Example
# =========================
def ai_code_generator_marker_search():
    """
    Example: Search for repositories containing AI code generator marker files (e.g., .claude, .cursor, .copilot, etc.).
    Prints a summary of repositories/files found for each marker and writes all results to a JSON file.
    Output file: ai_code_generator_analysis/ai_code_generator_markers.json
    """
    from github_api_scraper import GitHubAPIScraper
    import os
    import json
    from pathlib import Path
    
    # Get GitHub token from environment (recommended)
    github_token = os.getenv('GITHUB_TOKEN')
    scraper = GitHubAPIScraper(github_token)
    
    # Load existing data if available
    existing_data = None
    json_path = Path("ai_code_generator_analysis/ai_code_generator_markers.json")
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            print(f"Loaded {sum(len(hits) for hits in existing_data.values())} existing results")
        except Exception as e:
            print(f"Error loading existing data: {e}")
    
    # Search for AI code generator marker files (skipping existing ones)
    print("=== AI Code Generator Marker File Search ===")
    results = scraper.search_ai_code_generator_files(
        max_repos_per_pattern=10, 
        min_stars=0,
        existing_data=existing_data
    )
    
    for marker, hits in results.items():
        print(f"\nMarker: {marker}")
        if not hits:
            print("  No repositories found.")
        for hit in hits:
            print(f"  Repo: {hit['repo_name']} | File: {hit['file_path']} | Stars: {hit['stars']}")
            print(f"    Repo URL: {hit['repo_url']}")
            print(f"    File URL: {hit['file_url']}")
            if hit['description']:
                print(f"    Description: {hit['description']}")
    
    # Merge with existing data and write to JSON file
    output_dir = Path("ai_code_generator_analysis")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "ai_code_generator_markers.json"
    
    # Merge new results with existing data
    if existing_data:
        for marker, hits in results.items():
            if marker in existing_data:
                existing_data[marker].extend(hits)
            else:
                existing_data[marker] = hits
        final_results = existing_data
    else:
        final_results = results
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    total_new = sum(len(hits) for hits in results.values())
    total_final = sum(len(hits) for hits in final_results.values())
    print(f"\nAdded {total_new} new results")
    print(f"Total results in file: {total_final}")
    print(f"All results written to {output_file}")

# =========================
# Main Entry Point
# =========================
if __name__ == "__main__":
    print("GitHub API-Only Code Analyzer - Usage Examples")
    print("=" * 50)
    print("Available examples:")
    print("1. basic_usage() - Simple analysis")
    print("2. advanced_search_examples() - Different search strategies")
    print("3. analyze_specific_repositories() - Analyze specific repos")
    print("4. language_specific_analysis() - Focus on specific languages")
    print("5. batch_analysis_with_different_settings() - Multiple analyses")
    print("6. custom_suspicious_patterns() - Custom pattern detection")
    print("7. rate_limit_aware_analysis() - Rate limit management")
    print("8. export_and_visualization_example() - Export options")
    print("9. quick_repo_check() - Interactive single repo check")
    print("10. ai_code_generator_marker_search() - Search for AI code generator marker files")
    print("\nUncomment the function you want to run:")
    print()
    
    # Uncomment the example you want to run:
    # basic_usage()
    # advanced_search_examples()
    # analyze_specific_repositories()
    # language_specific_analysis()
    # batch_analysis_with_different_settings()
    # custom_suspicious_patterns()
    # rate_limit_aware_analysis()
    # export_and_visualization_example()
    # quick_repo_check()
    ai_code_generator_marker_search()
