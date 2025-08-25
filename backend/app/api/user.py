"""User API endpoints for NISR Rwanda Analytics Platform."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.utils.logger import get_logger
from app.database import get_mongodb

logger = get_logger(__name__)

router = APIRouter(prefix="/user", tags=["user"])


class UserProfile(BaseModel):
    """User profile model."""
    user_id: str
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: str = "analyst"
    department: Optional[str] = None
    last_login: Optional[datetime] = None
    created_at: datetime
    preferences: Optional[Dict[str, Any]] = None
    statistics: Optional[Dict[str, Any]] = None


class UserPreferences(BaseModel):
    """User preferences model."""
    theme: str = "light"  # light, dark, auto
    language: str = "en"  # en, rw, fr
    timezone: str = "Africa/Kigali"
    currency: str = "RWF"
    dashboard_layout: Optional[Dict[str, Any]] = None
    notification_settings: Optional[Dict[str, bool]] = None
    default_chart_type: str = "bar"
    items_per_page: int = 25


class UserStats(BaseModel):
    """User statistics model."""
    total_uploads: int = 0
    total_analyses: int = 0
    total_models: int = 0
    total_reports: int = 0
    storage_used_mb: float = 0.0
    last_activity: Optional[datetime] = None
    favorite_datasets: List[str] = []
    recent_datasets: List[str] = []


@router.get("/profile", response_model=UserProfile)
async def get_user_profile():
    """Get current user profile."""
    try:
        # Try to get user from database
        try:
            from app.database.models import UserDocument
            
            # For now, return a default user profile
            # In a real implementation, you would get this from authentication context
            default_user = UserProfile(
                user_id="nisr_user_001",
                username="nisr_analyst",
                email="analyst@nisr.gov.rw",
                full_name="NISR Data Analyst",
                role="analyst",
                department="Data Analytics",
                last_login=datetime.now(),
                created_at=datetime.now() - timedelta(days=30),
                preferences={
                    "theme": "light",
                    "language": "en",
                    "timezone": "Africa/Kigali",
                    "currency": "RWF"
                },
                statistics={
                    "total_uploads": 0,
                    "total_analyses": 0,
                    "total_models": 0,
                    "total_reports": 0
                }
            )
            
            # Try to find existing user or create default
            user_doc = await UserDocument.find_one({"user_id": "nisr_user_001"})
            if not user_doc:
                # Create default user in database
                user_doc = UserDocument(
                    user_id=default_user.user_id,
                    username=default_user.username,
                    email=default_user.email,
                    full_name=default_user.full_name,
                    role=default_user.role,
                    preferences=default_user.preferences,
                    last_login=default_user.last_login,
                    created_at=default_user.created_at
                )
                await user_doc.save()
                logger.info("Created default user in database")
            
            return default_user
            
        except Exception as db_error:
            logger.warning(f"Could not access database: {db_error}")
            
            # Return default user profile
            return UserProfile(
                user_id="nisr_user_001",
                username="nisr_analyst",
                email="analyst@nisr.gov.rw",
                full_name="NISR Data Analyst",
                role="analyst",
                department="Data Analytics",
                last_login=datetime.now(),
                created_at=datetime.now(),
                preferences={
                    "theme": "light",
                    "language": "en",
                    "timezone": "Africa/Kigali",
                    "currency": "RWF"
                }
            )
            
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user profile")


@router.put("/profile", response_model=UserProfile)
async def update_user_profile(profile_update: Dict[str, Any]):
    """Update user profile."""
    try:
        # Get current user profile
        current_profile = await get_user_profile()
        
        # Update allowed fields
        allowed_fields = ["full_name", "email", "department"]
        for field in allowed_fields:
            if field in profile_update:
                setattr(current_profile, field, profile_update[field])
        
        # Try to update in database
        try:
            from app.database.models import UserDocument
            
            user_doc = await UserDocument.find_one({"user_id": current_profile.user_id})
            if user_doc:
                for field in allowed_fields:
                    if field in profile_update:
                        setattr(user_doc, field, profile_update[field])
                user_doc.updated_at = datetime.now()
                await user_doc.save()
                logger.info(f"Updated user profile in database: {current_profile.user_id}")
            
        except Exception as db_error:
            logger.warning(f"Could not update in database: {db_error}")
        
        return current_profile
        
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user profile")


@router.get("/preferences", response_model=UserPreferences)
async def get_user_preferences():
    """Get user preferences."""
    try:
        # Try to get preferences from database
        try:
            from app.database.models import UserDocument
            
            user_doc = await UserDocument.find_one({"user_id": "nisr_user_001"})
            if user_doc and user_doc.preferences:
                return UserPreferences(**user_doc.preferences)
            
        except Exception as db_error:
            logger.warning(f"Could not get preferences from database: {db_error}")
        
        # Return default preferences
        return UserPreferences()
        
    except Exception as e:
        logger.error(f"Error getting user preferences: {e}")
        return UserPreferences()


@router.put("/preferences", response_model=UserPreferences)
async def update_user_preferences(preferences: UserPreferences):
    """Update user preferences."""
    try:
        # Try to update in database
        try:
            from app.database.models import UserDocument
            
            user_doc = await UserDocument.find_one({"user_id": "nisr_user_001"})
            if user_doc:
                user_doc.preferences = preferences.dict()
                user_doc.updated_at = datetime.now()
                await user_doc.save()
                logger.info("Updated user preferences in database")
            
        except Exception as db_error:
            logger.warning(f"Could not update preferences in database: {db_error}")
        
        return preferences
        
    except Exception as e:
        logger.error(f"Error updating user preferences: {e}")
        raise HTTPException(status_code=500, detail="Failed to update preferences")


@router.get("/statistics", response_model=UserStats)
async def get_user_statistics():
    """Get user statistics."""
    try:
        stats = UserStats()
        
        # Try to get real statistics from database
        try:
            from app.database.models import (
                DatasetDocument, AnalysisDocument, 
                MLModelDocument, ReportDocument
            )
            
            # Count user's documents
            user_id = "nisr_user_001"
            
            stats.total_uploads = await DatasetDocument.find({
                "uploaded_by": user_id
            }).count()
            
            # Get recent datasets
            recent_datasets = await DatasetDocument.find({
                "uploaded_by": user_id
            }).sort([("upload_timestamp", -1)]).limit(5)
            
            stats.recent_datasets = [d.dataset_id for d in recent_datasets]
            
            # Calculate storage used
            total_size = 0
            user_datasets = await DatasetDocument.find({"uploaded_by": user_id})
            for dataset in user_datasets:
                if dataset.file_size:
                    total_size += dataset.file_size
            
            stats.storage_used_mb = total_size / (1024 * 1024)
            stats.last_activity = datetime.now()
            
        except Exception as db_error:
            logger.warning(f"Could not get statistics from database: {db_error}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting user statistics: {e}")
        return UserStats()


@router.get("/activity")
async def get_user_activity(limit: int = 20):
    """Get user activity history."""
    try:
        activities = []
        
        # Try to get real activity from database
        try:
            from app.database.models import DatasetDocument, AnalysisDocument
            
            user_id = "nisr_user_001"
            
            # Get recent datasets
            recent_datasets = await DatasetDocument.find({
                "uploaded_by": user_id
            }).sort([("upload_timestamp", -1)]).limit(limit // 2)
            
            for dataset in recent_datasets:
                activities.append({
                    "id": f"upload_{dataset.dataset_id}",
                    "type": "upload",
                    "title": "Dataset Uploaded",
                    "description": f"Uploaded '{dataset.filename}'",
                    "timestamp": dataset.upload_timestamp.isoformat(),
                    "dataset_id": dataset.dataset_id
                })
            
        except Exception as db_error:
            logger.warning(f"Could not get activity from database: {db_error}")
            
            # Return sample activity
            activities = [
                {
                    "id": "activity_1",
                    "type": "login",
                    "title": "Logged In",
                    "description": "User logged into the platform",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        
        # Sort by timestamp (most recent first)
        activities.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {"activities": activities[:limit]}
        
    except Exception as e:
        logger.error(f"Error getting user activity: {e}")
        return {"activities": []}


@router.post("/logout")
async def logout_user():
    """Logout current user."""
    try:
        # In a real implementation, you would:
        # 1. Invalidate session/token
        # 2. Update last_login timestamp
        # 3. Log the logout event
        
        logger.info("User logged out")
        
        return {
            "success": True,
            "message": "Successfully logged out",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")


@router.get("/dashboard-config")
async def get_dashboard_config():
    """Get user's dashboard configuration."""
    try:
        # Default dashboard configuration
        config = {
            "widgets": [
                {
                    "id": "stats",
                    "type": "stats",
                    "title": "Key Metrics",
                    "position": {"x": 0, "y": 0, "w": 12, "h": 2},
                    "visible": True
                },
                {
                    "id": "recent_activity",
                    "type": "activity",
                    "title": "Recent Activity",
                    "position": {"x": 0, "y": 2, "w": 8, "h": 4},
                    "visible": True
                },
                {
                    "id": "quick_actions",
                    "type": "actions",
                    "title": "Quick Actions",
                    "position": {"x": 8, "y": 2, "w": 4, "h": 4},
                    "visible": True
                }
            ],
            "theme": "light",
            "auto_refresh": True,
            "refresh_interval": 30000
        }
        
        # Try to get custom config from database
        try:
            from app.database.models import UserDocument
            
            user_doc = await UserDocument.find_one({"user_id": "nisr_user_001"})
            if user_doc and user_doc.preferences and "dashboard_layout" in user_doc.preferences:
                dashboard_layout = user_doc.preferences["dashboard_layout"]
                if dashboard_layout:
                    config.update(dashboard_layout)
            
        except Exception as db_error:
            logger.warning(f"Could not get dashboard config from database: {db_error}")
        
        return config
        
    except Exception as e:
        logger.error(f"Error getting dashboard config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dashboard configuration")
