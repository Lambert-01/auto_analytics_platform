"""MongoDB database configuration and connection management."""

import asyncio
import os
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from pymongo.errors import ConnectionFailure

from app.utils.logger import get_logger
from app.database.models import *  # Import all Beanie models

logger = get_logger(__name__)


class MongoDB:
    """MongoDB connection manager."""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = None
        self.connection_url = self._get_mongodb_url()
        self.database_name = os.getenv("MONGODB_DATABASE", "nisr_analytics")
    
    def _get_mongodb_url(self) -> str:
        """Get MongoDB connection URL from environment variables."""
        
        # Check for full MongoDB URL
        mongodb_url = os.getenv("MONGODB_URL")
        if mongodb_url:
            return mongodb_url
        
        # Build URL from components
        host = os.getenv("MONGODB_HOST", "localhost")
        port = os.getenv("MONGODB_PORT", "27017")
        username = os.getenv("MONGODB_USERNAME")
        password = os.getenv("MONGODB_PASSWORD")
        
        if username and password:
            return f"mongodb://{username}:{password}@{host}:{port}"
        else:
            return f"mongodb://{host}:{port}"
    
    async def connect(self):
        """Connect to MongoDB and initialize Beanie."""
        try:
            logger.info(f"Connecting to MongoDB at: {self.connection_url}")
            
            # Create MongoDB client
            self.client = AsyncIOMotorClient(self.connection_url)
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info("MongoDB connection successful")
            
            # Get database
            self.database = self.client[self.database_name]
            
            # Initialize Beanie with document models
            await init_beanie(
                database=self.database,
                document_models=[
                    DatasetDocument,
                    AnalysisDocument, 
                    MLModelDocument,
                    ReportDocument,
                    ChatSessionDocument,
                    UserDocument,
                    RwandaProvinceDocument,
                    RwandaDistrictDocument
                ]
            )
            
            logger.info("Beanie ODM initialized successfully")
            
            # Create indexes for better performance
            await self._create_indexes()
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing MongoDB: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    async def _create_indexes(self):
        """Create database indexes for better performance."""
        try:
            # Dataset indexes
            await DatasetDocument.create_index("dataset_id")
            await DatasetDocument.create_index("filename")
            await DatasetDocument.create_index("upload_timestamp")
            
            # Analysis indexes
            await AnalysisDocument.create_index("analysis_id")
            await AnalysisDocument.create_index("dataset_id")
            await AnalysisDocument.create_index("created_at")
            
            # ML Model indexes
            await MLModelDocument.create_index("model_id")
            await MLModelDocument.create_index("dataset_id")
            await MLModelDocument.create_index("task_type")
            await MLModelDocument.create_index("status")
            
            # Report indexes
            await ReportDocument.create_index("report_id")
            await ReportDocument.create_index("report_type")
            await ReportDocument.create_index("created_at")
            
            # Chat session indexes
            await ChatSessionDocument.create_index("session_id")
            await ChatSessionDocument.create_index("created_at")
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Some indexes may not have been created: {e}")
    
    async def health_check(self) -> bool:
        """Check if MongoDB connection is healthy."""
        try:
            if not self.client:
                return False
            
            # Ping the database
            await self.client.admin.command('ping')
            return True
            
        except Exception:
            return False
    
    async def get_database_stats(self) -> dict:
        """Get database statistics."""
        try:
            if not self.database:
                return {}
            
            stats = await self.database.command("dbStats")
            
            # Get collection counts
            collections = {}
            for model in [DatasetDocument, AnalysisDocument, MLModelDocument, 
                         ReportDocument, ChatSessionDocument, UserDocument]:
                try:
                    count = await model.count()
                    collections[model.__name__] = count
                except Exception:
                    collections[model.__name__] = 0
            
            return {
                "database_name": self.database_name,
                "collections": collections,
                "data_size_mb": round(stats.get("dataSize", 0) / 1024 / 1024, 2),
                "storage_size_mb": round(stats.get("storageSize", 0) / 1024 / 1024, 2),
                "index_size_mb": round(stats.get("indexSize", 0) / 1024 / 1024, 2),
                "objects": stats.get("objects", 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}


# Global MongoDB instance
mongodb = MongoDB()


async def get_mongodb() -> MongoDB:
    """Get MongoDB instance."""
    return mongodb


async def init_database():
    """Initialize database connection."""
    await mongodb.connect()


async def close_database():
    """Close database connection."""
    await mongodb.disconnect()
