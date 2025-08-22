# 🇷🇼 NISR Rwanda Analytics Platform - Complete Setup Guide

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Database Configuration](#database-configuration)
4. [Application Installation](#application-installation)
5. [Frontend Setup](#frontend-setup)
6. [Production Deployment](#production-deployment)
7. [NISR-Specific Configuration](#nisr-specific-configuration)
8. [Troubleshooting](#troubleshooting)

---

## 🛠️ Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS 10.15+
- **Python**: 3.9 or higher
- **Node.js**: 16.0 or higher (for frontend development)
- **Memory**: Minimum 4GB RAM, Recommended 8GB+
- **Storage**: Minimum 10GB free space

### Required Software

#### 1. Python Environment
```bash
# Check Python version
python --version  # Should be 3.9+

# Install pip and virtualenv
python -m pip install --upgrade pip
python -m pip install virtualenv
```

#### 2. Database (Choose One)

**Option A: PostgreSQL (Recommended for Production)**
```bash
# Windows (using chocolatey)
choco install postgresql

# Ubuntu
sudo apt update
sudo apt install postgresql postgresql-contrib

# macOS
brew install postgresql
```

**Option B: SQLite (Development Only)**
- No installation required, built into Python

#### 3. Redis (For Caching and Sessions)
```bash
# Windows
choco install redis-64

# Ubuntu
sudo apt install redis-server

# macOS
brew install redis
```

---

## 🔧 Environment Setup

### 1. Clone and Setup Project
```bash
# Navigate to your project directory
cd auto_analytics_platform

# Create virtual environment
python -m virtualenv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Environment Configuration
```bash
# Copy environment template
cp config/environment.example .env

# Edit environment file
# Windows
notepad .env
# Linux/macOS
nano .env
```

### 3. Essential Environment Variables
```bash
# Application Settings
APP_NAME="NISR Rwanda Analytics Platform"
DEBUG=false
ENVIRONMENT=production

# Database Configuration (PostgreSQL)
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=nisr_analytics
DATABASE_USER=nisr_user
DATABASE_PASSWORD=your_secure_password

# Security
SECRET_KEY=your-super-secret-key-change-this-in-production
JWT_SECRET=your-jwt-secret-key

# File Storage
UPLOAD_DIR=./data/uploads
PROCESSED_DIR=./data/processed
MODELS_DIR=./data/models

# NISR Specific
ENABLE_CENSUS_MODULE=true
ENABLE_ECONOMIC_MODULE=true
VALIDATE_GEOGRAPHIC_CODES=true
RWANDA_PROVINCES=Kigali,Eastern,Northern,Southern,Western
```

---

## 🗄️ Database Configuration

### PostgreSQL Setup

#### 1. Install and Configure PostgreSQL
```bash
# Start PostgreSQL service
# Windows
net start postgresql-x64-14
# Ubuntu
sudo systemctl start postgresql
# macOS
brew services start postgresql
```

#### 2. Create Database and User
```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE nisr_analytics;

# Create user
CREATE USER nisr_user WITH PASSWORD 'your_secure_password';

# Grant permissions
GRANT ALL PRIVILEGES ON DATABASE nisr_analytics TO nisr_user;
GRANT CONNECT ON DATABASE nisr_analytics TO nisr_user;

# Exit
\q
```

#### 3. Run Database Setup Script
```bash
# Make script executable (Linux/macOS)
chmod +x scripts/setup_database.py

# Run setup script
python scripts/setup_database.py
```

### SQLite Setup (Development Only)
```bash
# No setup required - database will be created automatically
# File location: ./nisr_analytics.db
```

---

## 🚀 Application Installation

### 1. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Initialize database
python -c "from app.config import settings; settings.create_directories()"

# Run database migrations (if using Alembic)
alembic upgrade head
```

### 2. Start Backend Server
```bash
# Development mode
python start.py

# Production mode with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8000
```

### 3. Verify Backend Installation
```bash
# Check health endpoint
curl http://localhost:8000/health

# Check API documentation
# Open browser: http://localhost:8000/docs
```

---

## 🎨 Frontend Setup

### 1. Static Files (Current Setup)
The frontend is already configured with static HTML, CSS, and JavaScript files:

```
frontend/
├── static/
│   ├── css/
│   │   ├── main.css
│   │   ├── components.css
│   │   ├── responsive.css
│   │   ├── advanced.css
│   │   └── sidebar.css
│   ├── js/
│   │   ├── main.js
│   │   ├── api-client.js
│   │   ├── sidebar-navigation.js
│   │   ├── real-time.js
│   │   └── utils.js
│   └── images/
└── templates/
    ├── layout.html
    └── dashboard.html
```

### 2. Access Application
```bash
# Open in browser
http://localhost:8000
```

### 3. Login Credentials
```
Username: admin
Password: admin123
Email: admin@nisr.gov.rw
```

---

## 🌐 Production Deployment

### 1. Server Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 100GB+ SSD
- **Network**: High-speed internet connection

### 2. Ubuntu Server Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install python3.9 python3.9-venv python3-pip nginx postgresql redis-server -y

# Create application user
sudo useradd -m -s /bin/bash nisr-analytics
sudo usermod -aG sudo nisr-analytics

# Switch to application user
sudo su - nisr-analytics
```

### 3. Application Deployment
```bash
# Clone repository
git clone <your-repository-url> /home/nisr-analytics/app
cd /home/nisr-analytics/app

# Setup virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Setup environment
cp config/environment.example .env
nano .env  # Configure for production
```

### 4. Nginx Configuration
```bash
# Create Nginx configuration
sudo nano /etc/nginx/sites-available/nisr-analytics

# Add configuration:
server {
    listen 80;
    server_name your-domain.nisr.gov.rw;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /static/ {
        alias /home/nisr-analytics/app/frontend/static/;
        expires 30d;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/nisr-analytics /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 5. Systemd Service
```bash
# Create service file
sudo nano /etc/systemd/system/nisr-analytics.service

# Add content:
[Unit]
Description=NISR Analytics Platform
After=network.target

[Service]
User=nisr-analytics
Group=nisr-analytics
WorkingDirectory=/home/nisr-analytics/app
Environment=PATH=/home/nisr-analytics/app/venv/bin
ExecStart=/home/nisr-analytics/app/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8000
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start service
sudo systemctl enable nisr-analytics
sudo systemctl start nisr-analytics
sudo systemctl status nisr-analytics
```

### 6. SSL Certificate (Let's Encrypt)
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Obtain certificate
sudo certbot --nginx -d your-domain.nisr.gov.rw

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

---

## 🇷🇼 NISR-Specific Configuration

### 1. Rwanda Administrative Data
```python
# Load Rwanda administrative boundaries
python scripts/load_rwanda_data.py

# This loads:
# - Provinces (5)
# - Districts (30)
# - Sectors (416)
# - Cells (2148)
# - Villages (14,837)
```

### 2. Census Data Integration
```bash
# Environment variables for census
ENABLE_CENSUS_MODULE=true
CENSUS_DATA_PATH=./data/census
CENSUS_API_KEY=your_census_api_key
```

### 3. Economic Indicators Setup
```bash
# Rwanda National Bank integration
BNR_API_KEY=your_bnr_api_key
BNR_API_URL=https://www.bnr.rw/statistics/

# World Bank API
WORLD_BANK_API_URL=https://api.worldbank.org/v2/

# Enable economic module
ENABLE_ECONOMIC_MODULE=true
```

### 4. Survey Data Configuration
```bash
# Demographic Health Survey
DHS_API_KEY=your_dhs_key

# Living Standards Measurement Study
LSMS_DATA_ACCESS=true

# Enable survey analysis
ENABLE_SURVEY_MODULE=true
```

### 5. Geographic Data Setup
```bash
# Enable PostGIS for geographic analysis
sudo apt install postgis postgresql-13-postgis-3

# In PostgreSQL
CREATE EXTENSION postgis;
CREATE EXTENSION postgis_topology;

# Configure in environment
ENABLE_GEO_ANALYSIS=true
GEO_DATA_PATH=./data/geo
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Database Connection Errors
```bash
# Check PostgreSQL service
sudo systemctl status postgresql

# Check connection
psql -h localhost -U nisr_user -d nisr_analytics

# Reset password
sudo -u postgres psql
ALTER USER nisr_user PASSWORD 'new_password';
```

#### 2. Permission Errors
```bash
# Fix file permissions
sudo chown -R nisr-analytics:nisr-analytics /home/nisr-analytics/app
sudo chmod -R 755 /home/nisr-analytics/app
```

#### 3. Port Already in Use
```bash
# Find process using port 8000
sudo netstat -tulpn | grep :8000
# or
sudo lsof -i :8000

# Kill process
sudo kill -9 <process_id>
```

#### 4. Memory Issues
```bash
# Check memory usage
free -h

# Check Python processes
ps aux | grep python

# Restart services if needed
sudo systemctl restart nisr-analytics
```

#### 5. Nginx Issues
```bash
# Check Nginx status
sudo systemctl status nginx

# Check configuration
sudo nginx -t

# View error logs
sudo tail -f /var/log/nginx/error.log
```

### Performance Optimization

#### 1. Database Optimization
```sql
-- Analyze database performance
ANALYZE;

-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
```

#### 2. Application Optimization
```bash
# Monitor application performance
htop

# Check disk usage
df -h

# Monitor logs
tail -f logs/auto_analytics.log
```

---

## 📞 Support and Maintenance

### Support Contacts
- **Technical Support**: analytics-support@nisr.gov.rw
- **Data Questions**: data-manager@nisr.gov.rw
- **System Issues**: it-support@nisr.gov.rw

### Documentation Links
- **API Documentation**: http://your-domain/docs
- **User Guide**: https://docs.nisr.gov.rw/analytics
- **NISR Data Portal**: https://data.nisr.gov.rw

### Maintenance Schedule
- **Daily**: Automated backups, log rotation
- **Weekly**: Performance monitoring, security updates
- **Monthly**: Capacity planning, user management
- **Quarterly**: Full system review, compliance audit

---

## 🎯 Quick Start Checklist

- [ ] Install Prerequisites (Python, PostgreSQL, Redis)
- [ ] Clone repository and setup virtual environment
- [ ] Configure environment variables (.env file)
- [ ] Setup database (PostgreSQL + run setup script)
- [ ] Install Python dependencies
- [ ] Start backend server
- [ ] Verify installation (health check + login)
- [ ] Configure NISR-specific modules
- [ ] Load Rwanda administrative data
- [ ] Setup production deployment (if applicable)
- [ ] Configure monitoring and backups

---

**🎉 Congratulations! Your NISR Rwanda Analytics Platform is now ready to transform data into actionable insights for Rwanda's development.**

For additional help, please refer to the documentation or contact the NISR IT support team.
