# AI Code Generator Marker API

A FastAPI-based web application that scrapes GitHub repositories for AI code generator markers and provides a REST API to query the collected data.

## Features

- **GitHub Scraping**: Automatically searches GitHub for repositories containing AI code generator markers
- **Contact Extraction**: Extracts email addresses and contact information from repository owners
- **REST API**: Full REST API with filtering, sorting, and pagination
- **Database Integration**: PostgreSQL database with SQLAlchemy ORM
- **Modern Frontend**: Clean, responsive web interface

## Tech Stack

- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL (Supabase)
- **Frontend**: HTML/CSS/JavaScript
- **Deployment**: Render (Backend) + Vercel (Frontend)

## API Endpoints

- `GET /api/health` - Health check with database connection test
- `GET /api/markers` - Get available markers
- `GET /api/owner_types` - Get owner types
- `GET /api/owner_logins` - Get owner logins
- `GET /api/contact-stats` - Get contact statistics
- `GET /api/hits` - Get hits with filtering, sorting, pagination
- `POST /api/run-scraper` - Run the scraper

## Local Development

### Prerequisites

- Python 3.8+
- PostgreSQL database (or Supabase)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AIScraper
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**
   ```bash
   # Create .env file
   DATABASE_URL=postgresql://username:password@localhost:5432/database
   GITHUB_TOKEN=your_github_token
   ```

4. **Run the application**
   ```bash
   uvicorn api.main:app --reload
   ```

5. **Access the application**
   - API: http://localhost:8000
   - Frontend: http://localhost:8000/public/index.html

## Deployment

### Backend (Render)

1. **Connect your GitHub repository to Render**
2. **Set environment variables in Render:**
   - `DATABASE_URL`: Your Supabase PostgreSQL connection string
   - `GITHUB_TOKEN`: Your GitHub API token

3. **Deploy using the `render.yaml` configuration**

### Frontend (Vercel)

1. **Deploy the `public/index.html` to Vercel**
2. **Update the API base URL in the frontend to point to your Render backend**

## Environment Variables

- `DATABASE_URL`: PostgreSQL connection string
- `GITHUB_TOKEN`: GitHub API token for scraping

## Database Schema

The application uses a single table `marker_hits` with the following columns:

- `id`: Primary key
- `marker`: The marker found (e.g., "ai-code-generator")
- `owner_type`: Repository owner type (User/Organization)
- `owner_login`: Repository owner username
- `repo_name`: Repository name
- `repo_url`: Repository URL
- `file_path`: Path to the file containing the marker
- `file_url`: URL to the file
- `stars`: Repository star count
- `forks`: Repository fork count
- `description`: Repository description
- `email`: Extracted email address
- `phone`: Extracted phone number
- `contact_source`: Source of contact information
- `contact_extracted_at`: Timestamp of contact extraction
- `latest_commit_date`: Latest commit date

## Security

- CORS is configured to only allow specific domains
- Environment variables are used for sensitive data
- Input validation on all API endpoints
- Rate limiting considerations for GitHub API

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.
