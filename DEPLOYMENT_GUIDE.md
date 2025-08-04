# Deployment Guide - Vercel + PostgreSQL

This guide explains how to deploy your AI Code Generator Scraper to Vercel with PostgreSQL.

## Prerequisites

1. **GitHub Account** - Your code should be in a GitHub repository
2. **Vercel Account** - Sign up at [vercel.com](https://vercel.com)
3. **PostgreSQL Database** - Choose one of the options below

## PostgreSQL Database Options

### Option 1: Vercel Postgres (Recommended)
- **Cost**: $20/month
- **Setup**: Integrated with Vercel deployment
- **Benefits**: Seamless integration, automatic environment variables

### Option 2: Supabase (Free Tier)
- **Cost**: Free (500MB database, 50MB file storage)
- **Setup**: Create account at [supabase.com](https://supabase.com)
- **Benefits**: Excellent dashboard, real-time features

### Option 3: Neon (Free Tier)
- **Cost**: Free (3GB storage, 10GB transfer)
- **Setup**: Create account at [neon.tech](https://neon.tech)
- **Benefits**: Serverless PostgreSQL, auto-scaling

## Deployment Steps

### Step 1: Prepare Your Repository

1. Ensure all files are committed to GitHub:
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

2. Verify these files are in your repository:
   - `backend.py`
   - `github_api_scraper.py`
   - `requirements.txt`
   - `vercel.json`
   - `modern_frontend.html`

### Step 2: Set Up PostgreSQL Database

#### If using Vercel Postgres:
1. Go to your Vercel dashboard
2. Create a new project from your GitHub repository
3. In the project settings, go to "Storage" tab
4. Create a new Postgres database
5. Vercel will automatically set the `DATABASE_URL` environment variable

#### If using Supabase:
1. Create account at [supabase.com](https://supabase.com)
2. Create a new project
3. Go to Settings > Database
4. Copy the connection string
5. Set as `DATABASE_URL` in Vercel environment variables

#### If using Neon:
1. Create account at [neon.tech](https://neon.tech)
2. Create a new project
3. Copy the connection string
4. Set as `DATABASE_URL` in Vercel environment variables

### Step 3: Deploy to Vercel

1. **Connect Repository**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository

2. **Configure Environment Variables**:
   - In project settings, go to "Environment Variables"
   - Add `DATABASE_URL` with your PostgreSQL connection string
   - Add `GITHUB_TOKEN` with your GitHub API token

3. **Deploy**:
   - Vercel will automatically detect the Python project
   - Click "Deploy"
   - Wait for build to complete

### Step 4: Migrate Data (Optional)

If you have existing SQLite data:

1. **Local Migration**:
   ```bash
   # Set your PostgreSQL DATABASE_URL
   export DATABASE_URL="postgresql://user:password@host:port/database"
   
   # Run migration script
   python migrate_to_postgresql.py
   ```

2. **Or Start Fresh**:
   - The application will create tables automatically
   - Run the scraper to collect new data

### Step 5: Test Your Deployment

1. **Check Health Endpoint**:
   ```
   https://your-app.vercel.app/health
   ```

2. **Test Scraper**:
   - Open your deployed frontend
   - Enter your GitHub token
   - Run the scraper

3. **Query Data**:
   ```
   https://your-app.vercel.app/hits
   ```

## Environment Variables

Set these in your Vercel project settings:

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `GITHUB_TOKEN` | GitHub API token | Yes (for scraper) |

## Troubleshooting

### Common Issues:

1. **Build Failures**:
   - Check that `requirements.txt` includes `psycopg2-binary`
   - Ensure all Python dependencies are listed

2. **Database Connection Errors**:
   - Verify `DATABASE_URL` is correct
   - Check if database is accessible from Vercel
   - Ensure database allows external connections

3. **Function Timeouts**:
   - Vercel has 10-second timeout for free tier
   - Consider upgrading to Pro plan for 50-second timeout
   - Optimize scraper to work within limits

4. **CORS Issues**:
   - Frontend and backend should be on same domain
   - Or configure CORS properly in `backend.py`

### Performance Optimization:

1. **Connection Pooling**: Already configured in the code
2. **Cold Starts**: Consider using Vercel Pro for better performance
3. **Database Indexes**: PostgreSQL will create indexes automatically

## Monitoring

1. **Vercel Dashboard**: Monitor function execution and errors
2. **Database Monitoring**: Use your database provider's dashboard
3. **Logs**: Check Vercel function logs for debugging

## Cost Considerations

### Vercel Pricing:
- **Free Tier**: 100GB bandwidth/month, 10-second function timeout
- **Pro Plan**: $20/month, 50-second timeout, more bandwidth

### Database Pricing:
- **Vercel Postgres**: $20/month (included with Pro)
- **Supabase**: Free tier available
- **Neon**: Free tier available

## Next Steps

1. **Custom Domain**: Configure in Vercel settings
2. **SSL Certificate**: Automatic with Vercel
3. **Monitoring**: Set up alerts for errors
4. **Backup**: Configure database backups
5. **Scaling**: Monitor usage and upgrade as needed 