#!/usr/bin/env python3
"""
Database Setup Script for NISR Rwanda Analytics Platform
Sets up PostgreSQL database with required tables, indexes, and sample data
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend path to Python path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

import asyncpg
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd

from app.config import settings


class DatabaseSetup:
    """Database setup and initialization."""
    
    def __init__(self):
        self.db_config = {
            'host': settings.database_host or 'localhost',
            'port': settings.database_port or 5432,
            'user': settings.database_user or 'postgres',
            'password': settings.database_password,
            'database': settings.database_name or 'nisr_analytics'
        }
        
    def create_database(self):
        """Create the database if it doesn't exist."""
        print("🗄️  Creating database...")
        
        try:
            # Connect to PostgreSQL server (not specific database)
            conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database='postgres'  # Connect to default database
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                (self.db_config['database'],)
            )
            exists = cursor.fetchone()
            
            if not exists:
                cursor.execute(f"CREATE DATABASE {self.db_config['database']}")
                print(f"✅ Database '{self.db_config['database']}' created successfully")
            else:
                print(f"✅ Database '{self.db_config['database']}' already exists")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error creating database: {e}")
            return False
        
        return True
    
    def create_user(self):
        """Create application user with appropriate permissions."""
        print("👤 Creating application user...")
        
        try:
            conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database='postgres'
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Create user if not exists
            app_user = 'nisr_analytics_user'
            app_password = 'nisr_secure_2024!'
            
            cursor.execute(
                f"SELECT 1 FROM pg_roles WHERE rolname = '{app_user}'"
            )
            exists = cursor.fetchone()
            
            if not exists:
                cursor.execute(
                    f"CREATE USER {app_user} WITH PASSWORD '{app_password}'"
                )
                cursor.execute(
                    f"GRANT CONNECT ON DATABASE {self.db_config['database']} TO {app_user}"
                )
                cursor.execute(
                    f"GRANT USAGE ON SCHEMA public TO {app_user}"
                )
                cursor.execute(
                    f"GRANT CREATE ON SCHEMA public TO {app_user}"
                )
                print(f"✅ User '{app_user}' created successfully")
                print(f"🔑 Password: {app_password}")
            else:
                print(f"✅ User '{app_user}' already exists")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error creating user: {e}")
            return False
        
        return True
    
    def create_tables(self):
        """Create application tables."""
        print("📊 Creating application tables...")
        
        try:
            engine = create_engine(settings.database_url_postgres)
            
            # Create tables SQL
            tables_sql = """
            -- Users and Authentication
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                hashed_password VARCHAR(255) NOT NULL,
                full_name VARCHAR(255),
                department VARCHAR(100),
                role VARCHAR(50) DEFAULT 'analyst',
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Datasets
            CREATE TABLE IF NOT EXISTS datasets (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                file_path VARCHAR(500),
                file_size BIGINT,
                file_type VARCHAR(50),
                rows_count INTEGER,
                columns_count INTEGER,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                uploaded_by INTEGER REFERENCES users(id),
                status VARCHAR(50) DEFAULT 'uploaded',
                data_type VARCHAR(100), -- census, economic, survey, etc.
                geographic_level VARCHAR(50), -- national, provincial, district
                time_period VARCHAR(100),
                metadata JSONB,
                quality_score FLOAT,
                is_public BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Analysis Jobs
            CREATE TABLE IF NOT EXISTS analysis_jobs (
                id SERIAL PRIMARY KEY,
                dataset_id INTEGER REFERENCES datasets(id),
                user_id INTEGER REFERENCES users(id),
                job_type VARCHAR(100), -- eda, modeling, prediction
                status VARCHAR(50) DEFAULT 'pending',
                parameters JSONB,
                results JSONB,
                error_message TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- ML Models
            CREATE TABLE IF NOT EXISTS ml_models (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                model_type VARCHAR(100), -- classification, regression, clustering
                algorithm VARCHAR(100),
                dataset_id INTEGER REFERENCES datasets(id),
                user_id INTEGER REFERENCES users(id),
                model_path VARCHAR(500),
                performance_metrics JSONB,
                hyperparameters JSONB,
                feature_importance JSONB,
                training_time INTEGER,
                status VARCHAR(50) DEFAULT 'training',
                version INTEGER DEFAULT 1,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Reports
            CREATE TABLE IF NOT EXISTS reports (
                id SERIAL PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                description TEXT,
                report_type VARCHAR(100),
                format VARCHAR(50), -- pdf, html, word
                file_path VARCHAR(500),
                dataset_id INTEGER REFERENCES datasets(id),
                model_id INTEGER REFERENCES ml_models(id),
                user_id INTEGER REFERENCES users(id),
                generation_time INTEGER,
                status VARCHAR(50) DEFAULT 'generating',
                template_id VARCHAR(100),
                parameters JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Rwanda Administrative Units
            CREATE TABLE IF NOT EXISTS rwanda_admin_units (
                id SERIAL PRIMARY KEY,
                level VARCHAR(20) NOT NULL, -- province, district, sector, cell, village
                code VARCHAR(20) UNIQUE NOT NULL,
                name VARCHAR(255) NOT NULL,
                parent_code VARCHAR(20),
                population INTEGER,
                area_km2 FLOAT,
                coordinates JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Data Quality Checks
            CREATE TABLE IF NOT EXISTS data_quality_checks (
                id SERIAL PRIMARY KEY,
                dataset_id INTEGER REFERENCES datasets(id),
                check_type VARCHAR(100),
                status VARCHAR(50),
                score FLOAT,
                issues JSONB,
                recommendations JSONB,
                checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- System Logs
            CREATE TABLE IF NOT EXISTS system_logs (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                action VARCHAR(100),
                resource_type VARCHAR(100),
                resource_id INTEGER,
                details JSONB,
                ip_address INET,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- API Keys and External Integrations
            CREATE TABLE IF NOT EXISTS api_integrations (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                provider VARCHAR(100),
                api_key_encrypted TEXT,
                endpoint_url VARCHAR(500),
                is_active BOOLEAN DEFAULT TRUE,
                last_sync TIMESTAMP,
                sync_frequency VARCHAR(50),
                configuration JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            # Execute table creation
            with engine.connect() as conn:
                for statement in tables_sql.split(';'):
                    if statement.strip():
                        conn.execute(text(statement))
                conn.commit()
            
            print("✅ Tables created successfully")
            
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            return False
        
        return True
    
    def create_indexes(self):
        """Create database indexes for performance."""
        print("🔍 Creating database indexes...")
        
        try:
            engine = create_engine(settings.database_url_postgres)
            
            indexes_sql = """
            -- Performance indexes
            CREATE INDEX IF NOT EXISTS idx_datasets_upload_date ON datasets(upload_date);
            CREATE INDEX IF NOT EXISTS idx_datasets_status ON datasets(status);
            CREATE INDEX IF NOT EXISTS idx_datasets_data_type ON datasets(data_type);
            CREATE INDEX IF NOT EXISTS idx_datasets_user ON datasets(uploaded_by);
            
            CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status ON analysis_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_analysis_jobs_user ON analysis_jobs(user_id);
            CREATE INDEX IF NOT EXISTS idx_analysis_jobs_dataset ON analysis_jobs(dataset_id);
            
            CREATE INDEX IF NOT EXISTS idx_ml_models_status ON ml_models(status);
            CREATE INDEX IF NOT EXISTS idx_ml_models_type ON ml_models(model_type);
            CREATE INDEX IF NOT EXISTS idx_ml_models_user ON ml_models(user_id);
            CREATE INDEX IF NOT EXISTS idx_ml_models_active ON ml_models(is_active);
            
            CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status);
            CREATE INDEX IF NOT EXISTS idx_reports_type ON reports(report_type);
            CREATE INDEX IF NOT EXISTS idx_reports_user ON reports(user_id);
            
            CREATE INDEX IF NOT EXISTS idx_rwanda_admin_level ON rwanda_admin_units(level);
            CREATE INDEX IF NOT EXISTS idx_rwanda_admin_parent ON rwanda_admin_units(parent_code);
            
            CREATE INDEX IF NOT EXISTS idx_system_logs_user ON system_logs(user_id);
            CREATE INDEX IF NOT EXISTS idx_system_logs_action ON system_logs(action);
            CREATE INDEX IF NOT EXISTS idx_system_logs_date ON system_logs(created_at);
            
            -- Full-text search indexes
            CREATE INDEX IF NOT EXISTS idx_datasets_search ON datasets USING gin(to_tsvector('english', name || ' ' || COALESCE(description, '')));
            CREATE INDEX IF NOT EXISTS idx_reports_search ON reports USING gin(to_tsvector('english', title || ' ' || COALESCE(description, '')));
            """
            
            with engine.connect() as conn:
                for statement in indexes_sql.split(';'):
                    if statement.strip():
                        conn.execute(text(statement))
                conn.commit()
            
            print("✅ Indexes created successfully")
            
        except Exception as e:
            print(f"❌ Error creating indexes: {e}")
            return False
        
        return True
    
    def insert_rwanda_admin_data(self):
        """Insert Rwanda administrative units data."""
        print("🇷🇼 Inserting Rwanda administrative data...")
        
        try:
            engine = create_engine(settings.database_url_postgres)
            
            # Rwanda administrative units sample data
            admin_data = [
                # Provinces
                ('province', 'P01', 'Kigali', None, 1132686, 730.0),
                ('province', 'P02', 'Eastern', None, 2695394, 9458.0),
                ('province', 'P03', 'Northern', None, 1726370, 3276.0),
                ('province', 'P04', 'Southern', None, 2814728, 5963.0),
                ('province', 'P05', 'Western', None, 2471239, 5883.0),
                
                # Sample Districts (Kigali Province)
                ('district', 'D0101', 'Gasabo', 'P01', 429681, 430.0),
                ('district', 'D0102', 'Kicukiro', 'P01', 376829, 166.0),
                ('district', 'D0103', 'Nyarugenge', 'P01', 326176, 134.0),
                
                # Sample Sectors (Gasabo District)
                ('sector', 'S010101', 'Bumbogo', 'D0101', 65456, 52.0),
                ('sector', 'S010102', 'Gatsata', 'D0101', 82139, 45.0),
                ('sector', 'S010103', 'Jali', 'D0101', 31245, 38.0),
                ('sector', 'S010104', 'Kimisagara', 'D0101', 67890, 25.0),
            ]
            
            with engine.connect() as conn:
                # Clear existing data
                conn.execute(text("DELETE FROM rwanda_admin_units"))
                
                # Insert new data
                for level, code, name, parent_code, population, area in admin_data:
                    conn.execute(text("""
                        INSERT INTO rwanda_admin_units (level, code, name, parent_code, population, area_km2)
                        VALUES (:level, :code, :name, :parent_code, :population, :area)
                    """), {
                        'level': level,
                        'code': code,
                        'name': name,
                        'parent_code': parent_code,
                        'population': population,
                        'area': area
                    })
                
                conn.commit()
            
            print("✅ Rwanda administrative data inserted successfully")
            
        except Exception as e:
            print(f"❌ Error inserting Rwanda admin data: {e}")
            return False
        
        return True
    
    def create_sample_user(self):
        """Create a sample admin user."""
        print("👤 Creating sample admin user...")
        
        try:
            engine = create_engine(settings.database_url_postgres)
            
            # Simple password hashing (use proper hashing in production)
            import hashlib
            password = "admin123"
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            
            with engine.connect() as conn:
                # Check if user exists
                result = conn.execute(text(
                    "SELECT id FROM users WHERE username = :username"
                ), {'username': 'admin'})
                
                if not result.fetchone():
                    conn.execute(text("""
                        INSERT INTO users (username, email, hashed_password, full_name, department, role)
                        VALUES (:username, :email, :password, :full_name, :department, :role)
                    """), {
                        'username': 'admin',
                        'email': 'admin@nisr.gov.rw',
                        'password': hashed_password,
                        'full_name': 'NISR Administrator',
                        'department': 'Information Technology',
                        'role': 'admin'
                    })
                    conn.commit()
                    print("✅ Sample admin user created")
                    print("   Username: admin")
                    print("   Password: admin123")
                    print("   Email: admin@nisr.gov.rw")
                else:
                    print("✅ Admin user already exists")
            
        except Exception as e:
            print(f"❌ Error creating sample user: {e}")
            return False
        
        return True
    
    def setup_extensions(self):
        """Setup PostgreSQL extensions."""
        print("🔧 Setting up PostgreSQL extensions...")
        
        try:
            engine = create_engine(settings.database_url_postgres)
            
            extensions = [
                "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;",
                "CREATE EXTENSION IF NOT EXISTS pg_trgm;",
                "CREATE EXTENSION IF NOT EXISTS uuid-ossp;",
                "CREATE EXTENSION IF NOT EXISTS postgis;"  # For geographic data
            ]
            
            with engine.connect() as conn:
                for ext in extensions:
                    try:
                        conn.execute(text(ext))
                        print(f"   ✅ {ext.split()[4]} extension enabled")
                    except Exception as e:
                        print(f"   ⚠️  {ext.split()[4]} extension skipped: {e}")
                
                conn.commit()
            
        except Exception as e:
            print(f"❌ Error setting up extensions: {e}")
            return False
        
        return True
    
    def run_setup(self):
        """Run complete database setup."""
        print("🚀 Starting NISR Rwanda Analytics Platform Database Setup")
        print("=" * 60)
        
        steps = [
            ("Creating database", self.create_database),
            ("Creating user", self.create_user),
            ("Setting up extensions", self.setup_extensions),
            ("Creating tables", self.create_tables),
            ("Creating indexes", self.create_indexes),
            ("Inserting Rwanda admin data", self.insert_rwanda_admin_data),
            ("Creating sample user", self.create_sample_user),
        ]
        
        success_count = 0
        for step_name, step_func in steps:
            print(f"\n{step_name}...")
            if step_func():
                success_count += 1
            else:
                print(f"❌ Failed: {step_name}")
        
        print("\n" + "=" * 60)
        if success_count == len(steps):
            print("🎉 Database setup completed successfully!")
            print("\n📋 Next Steps:")
            print("1. Update your .env file with database credentials")
            print("2. Start the application server")
            print("3. Login with admin/admin123")
            print("4. Upload your first dataset")
        else:
            print(f"⚠️  Setup completed with {len(steps) - success_count} warnings/errors")
        
        print("\n🔗 Connection String:")
        print(f"   {settings.database_url_postgres}")


def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print("NISR Rwanda Analytics Platform Database Setup")
        print("Usage: python setup_database.py")
        print("\nEnvironment Variables:")
        print("  DATABASE_HOST     - PostgreSQL host (default: localhost)")
        print("  DATABASE_PORT     - PostgreSQL port (default: 5432)")
        print("  DATABASE_USER     - PostgreSQL user (default: postgres)")
        print("  DATABASE_PASSWORD - PostgreSQL password")
        print("  DATABASE_NAME     - Database name (default: nisr_analytics)")
        return
    
    setup = DatabaseSetup()
    setup.run_setup()


if __name__ == "__main__":
    main()