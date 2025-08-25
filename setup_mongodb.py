#!/usr/bin/env python3
"""
MongoDB Setup Script for NISR Rwanda Analytics Platform
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_step(step, description):
    """Print a formatted step."""
    print(f"\n🔹 Step {step}: {description}")

def run_command(command, description=""):
    """Run a shell command and handle errors."""
    try:
        print(f"   Running: {command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   Error details: {e.stderr}")
        return False

def check_mongodb_installation():
    """Check if MongoDB is installed."""
    try:
        result = subprocess.run(["mongod", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ MongoDB is installed")
            return True
    except FileNotFoundError:
        pass
    
    print("   ❌ MongoDB is not installed")
    return False

def install_mongodb_windows():
    """Guide for installing MongoDB on Windows."""
    print("""
   📋 MongoDB Installation Guide for Windows:
   
   1. Download MongoDB Community Server from:
      https://www.mongodb.com/try/download/community
   
   2. Run the installer (.msi file) and follow the setup wizard
   
   3. Choose "Complete" installation
   
   4. Install MongoDB as a Windows Service (recommended)
   
   5. Install MongoDB Compass (GUI tool) - optional but recommended
   
   6. Add MongoDB to your PATH:
      - Default installation path: C:\\Program Files\\MongoDB\\Server\\[version]\\bin
      - Add this to your system PATH environment variable
   
   7. Restart your command prompt/PowerShell
   
   8. Verify installation by running: mongod --version
   """)

def install_mongodb_macos():
    """Guide for installing MongoDB on macOS."""
    print("""
   📋 MongoDB Installation Guide for macOS:
   
   Using Homebrew (recommended):
   1. Install Homebrew if not already installed:
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   2. Add MongoDB tap:
      brew tap mongodb/brew
   
   3. Install MongoDB Community Edition:
      brew install mongodb-community
   
   4. Start MongoDB service:
      brew services start mongodb/brew/mongodb-community
   
   Manual Installation:
   1. Download from: https://www.mongodb.com/try/download/community
   2. Extract the archive
   3. Copy files to /usr/local/mongodb
   4. Add /usr/local/mongodb/bin to your PATH
   """)

def install_mongodb_linux():
    """Guide for installing MongoDB on Linux."""
    print("""
   📋 MongoDB Installation Guide for Linux:
   
   Ubuntu/Debian:
   1. Import the public key:
      wget -qO - https://www.mongodb.org/static/pgp/server-7.0.asc | sudo apt-key add -
   
   2. Create the list file:
      echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
   
   3. Update package database:
      sudo apt-get update
   
   4. Install MongoDB:
      sudo apt-get install -y mongodb-org
   
   5. Start MongoDB:
      sudo systemctl start mongod
      sudo systemctl enable mongod
   
   CentOS/RHEL/Fedora:
   1. Create repository file:
      sudo vi /etc/yum.repos.d/mongodb-org-7.0.repo
   
   2. Add repository configuration:
      [mongodb-org-7.0]
      name=MongoDB Repository
      baseurl=https://repo.mongodb.org/yum/redhat/$releasever/mongodb-org/7.0/x86_64/
      gpgcheck=1
      enabled=1
      gpgkey=https://www.mongodb.org/static/pgp/server-7.0.asc
   
   3. Install MongoDB:
      sudo yum install -y mongodb-org
   
   4. Start MongoDB:
      sudo systemctl start mongod
      sudo systemctl enable mongod
   """)

def setup_mongodb_database():
    """Set up MongoDB database and collections."""
    print_step("5", "Setting up MongoDB database and collections")
    
    try:
        import pymongo
        from pymongo import MongoClient
    except ImportError:
        print("   ❌ PyMongo not installed. Installing...")
        if not run_command("pip install pymongo"):
            return False
        import pymongo
        from pymongo import MongoClient
    
    try:
        # Connect to MongoDB
        client = MongoClient("mongodb://localhost:27017/")
        
        # Test connection
        client.admin.command('ping')
        print("   ✅ Successfully connected to MongoDB")
        
        # Create database
        db = client["nisr_analytics"]
        
        # Create collections and indexes
        collections = [
            "datasets", "analyses", "ml_models", "reports", 
            "chat_sessions", "users", "rwanda_provinces", "rwanda_districts"
        ]
        
        for collection_name in collections:
            collection = db[collection_name]
            # Create a simple index
            collection.create_index("created_at")
            print(f"   ✅ Created collection: {collection_name}")
        
        # Insert sample Rwanda provinces data
        provinces_data = [
            {"province_id": "prov_001", "province_name": "Kigali City", "province_code": "KIG"},
            {"province_id": "prov_002", "province_name": "Eastern Province", "province_code": "EST"},
            {"province_id": "prov_003", "province_name": "Northern Province", "province_code": "NOR"},
            {"province_id": "prov_004", "province_name": "Southern Province", "province_code": "SOU"},
            {"province_id": "prov_005", "province_name": "Western Province", "province_code": "WST"}
        ]
        
        db.rwanda_provinces.insert_many(provinces_data)
        print("   ✅ Inserted Rwanda provinces data")
        
        print("   ✅ MongoDB database setup completed")
        return True
        
    except Exception as e:
        print(f"   ❌ Error setting up MongoDB database: {e}")
        return False

def install_python_dependencies():
    """Install Python dependencies including MongoDB packages."""
    print_step("6", "Installing Python dependencies")
    
    # Install requirements
    if not run_command("pip install -r backend/requirements.txt"):
        print("   ❌ Failed to install requirements")
        return False
    
    print("   ✅ Python dependencies installed successfully")
    return True

def create_environment_file():
    """Create .env file with MongoDB configuration."""
    print_step("7", "Creating environment configuration")
    
    env_content = """# NISR Rwanda Analytics Platform Configuration

# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=nisr_analytics
MONGODB_HOST=localhost
MONGODB_PORT=27017
# MONGODB_USERNAME=your_username  # Uncomment if using authentication
# MONGODB_PASSWORD=your_password  # Uncomment if using authentication

# Application Settings
DEBUG=True
API_TITLE=NISR Rwanda Analytics Platform
API_VERSION=1.0.0
SECRET_KEY=your-super-secret-key-change-this-in-production

# NISR Specific Settings
APP_NAME=NISR Rwanda Analytics Platform
ENVIRONMENT=development
TIMEZONE=Africa/Kigali
CURRENCY=RWF
LANGUAGE=en
LOCALE=en_RW

# External APIs (Optional - add your keys)
# OPENAI_API_KEY=your_openai_api_key
# BNR_API_KEY=your_bnr_api_key
# WORLDBANK_API_KEY=your_worldbank_api_key

# Feature Flags
ENABLE_AI_FEATURES=True
ENABLE_AUTOML=True
ENABLE_REAL_TIME=True
ENABLE_CHAT_INTERFACE=True

# Performance Settings
MAX_UPLOAD_SIZE_MB=500
MAX_ANALYSIS_TIME_MINUTES=30
MAX_AUTOML_TIME_MINUTES=60
"""
    
    try:
        with open(".env", "w") as f:
            f.write(env_content)
        print("   ✅ Created .env file with MongoDB configuration")
        return True
    except Exception as e:
        print(f"   ❌ Error creating .env file: {e}")
        return False

def main():
    """Main setup function."""
    print_section("NISR Rwanda Analytics Platform - MongoDB Setup")
    print("🚀 Setting up MongoDB for your analytics platform...")
    
    # Step 1: Check system
    print_step("1", "Checking system requirements")
    current_os = sys.platform
    print(f"   Operating System: {current_os}")
    
    # Step 2: Check MongoDB installation
    print_step("2", "Checking MongoDB installation")
    mongodb_installed = check_mongodb_installation()
    
    if not mongodb_installed:
        print_step("3", "MongoDB Installation Required")
        if current_os.startswith("win"):
            install_mongodb_windows()
        elif current_os.startswith("darwin"):
            install_mongodb_macos()
        else:
            install_mongodb_linux()
        
        print("\n   ⚠️  Please install MongoDB and run this script again.")
        sys.exit(1)
    
    # Step 4: Check if MongoDB is running
    print_step("4", "Checking MongoDB service")
    try:
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
        client.admin.command('ping')
        print("   ✅ MongoDB is running")
    except Exception as e:
        print(f"   ❌ MongoDB is not running: {e}")
        print("\n   Please start MongoDB service:")
        if current_os.startswith("win"):
            print("   - Windows: Start MongoDB service from Services or run 'net start MongoDB'")
        elif current_os.startswith("darwin"):
            print("   - macOS: brew services start mongodb/brew/mongodb-community")
        else:
            print("   - Linux: sudo systemctl start mongod")
        sys.exit(1)
    
    # Step 5: Setup database
    if not setup_mongodb_database():
        sys.exit(1)
    
    # Step 6: Install dependencies
    if not install_python_dependencies():
        sys.exit(1)
    
    # Step 7: Create environment file
    if not create_environment_file():
        sys.exit(1)
    
    # Success
    print_section("Setup Complete! 🎉")
    print("""
✅ MongoDB setup completed successfully!

🔧 What was configured:
   • MongoDB database: nisr_analytics
   • Collections for datasets, analyses, models, reports, chat sessions
   • Sample Rwanda provinces data
   • Python dependencies (pymongo, motor, beanie)
   • Environment configuration (.env file)

🚀 Next steps:
   1. Start the platform: python start.py
   2. Open your browser: http://localhost:8000
   3. Upload data and start analyzing!

📋 MongoDB Management:
   • Database: nisr_analytics
   • Connection: mongodb://localhost:27017
   • GUI Tool: MongoDB Compass (if installed)

🔧 Configuration:
   • Edit .env file for custom settings
   • Modify backend/app/database/mongodb.py for advanced configuration

💡 Need help?
   • Check logs in logs/ directory
   • View API docs at http://localhost:8000/docs
   • MongoDB logs: mongod logs (varies by OS)
""")

if __name__ == "__main__":
    main()
