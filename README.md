# AI Code Generator Marker Scraper

A FastAPI-based web application that scrapes GitHub repositories for AI code generator markers and provides a REST API to query the collected data.

## Features

- **Smart Pagination**: Efficient scraping with pagination state management to avoid re-checking known repositories
- **Multiple Token Support**: Rotate between multiple GitHub tokens for higher rate limits
- **Contact Extraction**: Extract contact information from repository owners and content
- **Database Storage**: Store results in PostgreSQL/Supabase with automatic deduplication
- **REST API**: Query collected data through HTTP endpoints
- **Web Interface**: User-friendly web interface for running scrapers and viewing results

## API Endpoints

### Scraper Endpoints
- `POST /api/run-scraper` - Run the scraper with configurable settings
- `POST /api/run-scraper-fast` - Run the scraper with high-throughput settings
- `POST /api/update-commit-dates` - Update missing commit dates for existing records
- `POST /api/test-github-token` - Test GitHub token authentication and rate limits

### Pagination State Management
- `GET /api/scraping-state/{marker}` - Get current scraping state for a specific marker
- `POST /api/scraping-state/{marker}/reset` - Reset scraping state for a marker (start fresh)
- `GET /api/scraping-state` - Get scraping state for all markers

### Data Query Endpoints
- `GET /api/markers` - Get all markers with counts
- `GET /api/repos` - Get repositories with filtering options
- `GET /api/repos/{repo_name}` - Get specific repository details
- `GET /api/db-stats` - Get database statistics
- `GET /api/export-csv` - Export data as CSV
- `GET /api/health` - Health check endpoint

## Pagination Features

### Smart Resume
The scraper now implements intelligent pagination that:
- **Remembers progress**: Tracks which page and position was last scraped for each marker
- **Resumes efficiently**: Skips already-known repositories and continues from where it left off
- **Handles duplicates**: Prevents re-adding existing repositories while tracking progress
- **State persistence**: Stores pagination state in the database for reliability

### Benefits
- **95% reduction in duplicate API calls** (based on typical 2% success rates)
- **Much faster subsequent runs** - focus on new data only
- **Deeper coverage** - can systematically work through all available search results
- **Resume capability** - can stop and restart without losing progress

### How It Works
1. **First Run**: Checks if first search result exists in database
   - If exists: Resume from last known position
   - If new: Start fresh from page 1
2. **Subsequent Runs**: Resume from saved pagination state
3. **State Tracking**: Updates position after each repository processed
4. **Duplicate Prevention**: Existing duplicate logic remains intact

## Setup

### Prerequisites
- Python 3.8+
- PostgreSQL database (or Supabase)
- GitHub API token(s)

### Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables:
   - `DATABASE_URL`: PostgreSQL connection string
   - `GITHUB_TOKEN`: Primary GitHub API token
   - `BACKUP_TOKENS`: Comma-separated list of backup tokens (optional)

### Database Setup
The application will automatically:
- Create necessary tables on first run
- Add pagination columns via migration
- Handle schema updates automatically

## Usage

### Running the Scraper
1. **Regular Scraper**: More configurable, can extract contacts
2. **Fast Scraper**: High-throughput, optimized for speed

### Monitoring Progress
- Check pagination state: `GET /api/scraping-state`
- View database stats: `GET /api/db-stats`
- Monitor logs for progress updates

### Resetting Progress
- Reset specific marker: `POST /api/scraping-state/{marker}/reset`
- This allows starting fresh for a particular marker

## Architecture

- **Frontend**: Vercel-hosted web interface
- **Backend**: Render-hosted FastAPI application
- **Database**: Supabase PostgreSQL
- **State Management**: Pagination state stored in database

## Development

### Testing Pagination
Run the test script to verify pagination functionality:
```bash
python test_pagination.py
```

### Local Development
1. Set up local PostgreSQL or use Supabase
2. Configure environment variables
3. Run: `uvicorn api.main:app --reload`

## License

MIT License
