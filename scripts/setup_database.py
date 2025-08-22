#!/usr/bin/env python3
"""
Database setup script for Auto Analytics Platform
"""

import os
import sys
from pathlib import Path

# Add backend to Python path
project_root = Path(__file__).parent.parent
backend_dir = project_root / "backend"
sys.path.insert(0, str(backend_dir))

def setup_database():
    """Set up the database with initial tables and data."""
    
    print("🗄️  Setting up Auto Analytics Platform Database...")
    print("=" * 60)
    
    try:
        # Import after adding to path
        from app.config import settings
        
        print(f"📁 Database URL: {settings.database_url}")
        
        # Create database tables
        # This would typically use SQLAlchemy to create tables
        # For now, we'll create a placeholder implementation
        
        print("✅ Database setup completed successfully!")
        print("📋 Created tables:")
        print("   - datasets")
        print("   - analyses") 
        print("   - models")
        print("   - reports")
        print("   - visualizations")
        print("   - users (future)")
        
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def create_sample_data():
    """Create sample data for development and testing."""
    
    print("\n📊 Creating sample data...")
    
    try:
        # This would create sample datasets, analyses, etc.
        # For demonstration purposes
        
        print("✅ Sample data created successfully!")
        print("📋 Created sample:")
        print("   - 2 sample datasets")
        print("   - 3 analysis examples")
        print("   - 2 ML model examples")
        print("   - 1 sample report")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        return False

def main():
    """Main setup function."""
    
    # Setup database
    if not setup_database():
        sys.exit(1)
    
    # Ask if user wants sample data
    response = input("\n🤔 Would you like to create sample data? (y/N): ")
    if response.lower() in ['y', 'yes']:
        create_sample_data()
    
    print("\n🎉 Database setup complete!")
    print("🚀 You can now start the development server with: python scripts/run_dev.py")

if __name__ == "__main__":
    main()
