#!/usr/bin/env python3
"""
Development server runner for Auto Analytics Platform
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the development server with proper setup."""
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    backend_dir = project_root / "backend"
    
    # Change to backend directory
    os.chdir(backend_dir)
    
    # Add backend to Python path
    sys.path.insert(0, str(backend_dir))
    
    print("🚀 Starting Auto Analytics Platform Development Server...")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Backend directory: {backend_dir}")
    print("=" * 60)
    
    try:
        # Check if .env file exists
        env_file = project_root / ".env"
        if not env_file.exists():
            print("⚠️  No .env file found. Creating from .env.example...")
            example_file = project_root / ".env.example"
            if example_file.exists():
                import shutil
                shutil.copy(example_file, env_file)
                print("✅ Created .env file from .env.example")
            else:
                print("❌ No .env.example file found. Please create .env manually.")
        
        # Create necessary directories
        directories = [
            project_root / "data" / "uploads",
            project_root / "data" / "processed", 
            project_root / "data" / "models",
            project_root / "data" / "cache",
            project_root / "reports" / "html",
            project_root / "reports" / "pdf",
            project_root / "reports" / "templates",
            project_root / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✅ Created necessary directories")
        
        # Run the FastAPI server
        print("🔄 Starting FastAPI server...")
        print("📡 Server will be available at: http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🔍 Health Check: http://localhost:8000/health")
        print("=" * 60)
        
        # Use uvicorn to run the server
        cmd = [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--reload-dir", str(backend_dir),
            "--log-level", "info"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
