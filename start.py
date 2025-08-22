#!/usr/bin/env python3
"""
Quick start script for Auto Analytics Platform
"""

import os
import sys
from pathlib import Path

def main():
    """Quick start for the Auto Analytics Platform."""
    
    print("🚀 Auto Analytics Platform - Quick Start")
    print("=" * 50)
    
    # Get project root
    project_root = Path(__file__).parent
    backend_dir = project_root / "backend"
    
    # Create necessary directories
    directories = [
        "data/uploads", "data/processed", "data/models", "data/cache",
        "reports/html", "reports/pdf", "reports/templates", "logs"
    ]
    
    for directory in directories:
        (project_root / directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created necessary directories")
    
    # Check if we're in the correct directory
    if not backend_dir.exists():
        print("❌ Backend directory not found. Please run from project root.")
        sys.exit(1)
    
    # Change to backend directory and run
    os.chdir(backend_dir)
    sys.path.insert(0, str(backend_dir))
    
    print("🔄 Starting development server...")
    print("📡 Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔍 Health: http://localhost:8000/health")
    print("=" * 50)
    
    try:
        # Import and run the FastAPI app
        import uvicorn
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except ImportError:
        print("❌ Missing dependencies. Please install requirements:")
        print("   cd backend && pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")

if __name__ == "__main__":
    main()
