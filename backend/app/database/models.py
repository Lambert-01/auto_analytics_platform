"""Beanie document models for MongoDB."""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from beanie import Document, Indexed
from pydantic import Field
from bson import ObjectId


class DatasetStatus(str, Enum):
    """Dataset processing status."""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisStatus(str, Enum):
    """Analysis status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelStatus(str, Enum):
    """ML model status."""
    PENDING = "pending"
    TRAINING = "training" 
    COMPLETED = "completed"
    FAILED = "failed"
    DEPLOYED = "deployed"


class ReportType(str, Enum):
    """Report types."""
    DATASET_SUMMARY = "dataset_summary"
    ANALYSIS_REPORT = "analysis_report"
    ML_MODEL_REPORT = "ml_model_report"
    CUSTOM_REPORT = "custom_report"
    COMPREHENSIVE = "comprehensive"


class DatasetDocument(Document):
    """Dataset document for MongoDB."""
    
    dataset_id: Indexed(str, unique=True)
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    file_type: str
    
    # Dataset metadata
    rows: Optional[int] = None
    columns: Optional[int] = None
    column_names: Optional[List[str]] = None
    column_types: Optional[Dict[str, str]] = None
    
    # Processing information
    status: DatasetStatus = DatasetStatus.UPLOADING
    processing_logs: Optional[List[str]] = None
    error_message: Optional[str] = None
    
    # Data quality metrics
    missing_values: Optional[Dict[str, int]] = None
    duplicate_rows: Optional[int] = None
    data_quality_score: Optional[float] = None
    
    # Timestamps
    upload_timestamp: datetime = Field(default_factory=datetime.now)
    processing_start_time: Optional[datetime] = None
    processing_end_time: Optional[datetime] = None
    processing_duration: Optional[float] = None
    
    # User information
    uploaded_by: Optional[str] = None
    
    # NISR specific fields
    data_source: Optional[str] = None  # census, survey, economic, etc.
    collection_period: Optional[str] = None
    geographic_coverage: Optional[List[str]] = None  # provinces, districts
    
    class Settings:
        name = "datasets"
        indexes = [
            "dataset_id",
            "filename", 
            "upload_timestamp",
            "status",
            "data_source"
        ]


class AnalysisDocument(Document):
    """Analysis document for MongoDB."""
    
    analysis_id: Indexed(str, unique=True)
    dataset_id: str
    analysis_type: List[str]  # descriptive, correlation, clustering, etc.
    
    # Analysis configuration
    target_column: Optional[str] = None
    selected_columns: Optional[List[str]] = None
    analysis_parameters: Optional[Dict[str, Any]] = None
    
    # Results
    status: AnalysisStatus = AnalysisStatus.PENDING
    results: Optional[Dict[str, Any]] = None
    insights: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    
    # Visualizations
    visualizations: Optional[List[Dict[str, Any]]] = None
    charts_generated: Optional[int] = None
    
    # Performance metrics
    data_quality_score: Optional[float] = None
    execution_time: Optional[float] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Error handling
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    class Settings:
        name = "analyses"
        indexes = [
            "analysis_id",
            "dataset_id",
            "created_at",
            "status"
        ]


class MLModelDocument(Document):
    """ML Model document for MongoDB."""
    
    model_id: Indexed(str, unique=True)
    model_name: str
    dataset_id: str
    
    # Model configuration
    task_type: str  # classification, regression, clustering
    algorithm: str
    target_column: str
    feature_columns: List[str]
    
    # Training configuration
    training_config: Optional[Dict[str, Any]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    
    # Model status and performance
    status: ModelStatus = ModelStatus.PENDING
    training_progress: Optional[float] = None  # 0-100
    
    # Performance metrics
    metrics: Optional[Dict[str, Any]] = None
    cross_validation_scores: Optional[List[float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    
    # Model artifacts
    model_file_path: Optional[str] = None
    model_size_bytes: Optional[int] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    training_start_time: Optional[datetime] = None
    training_end_time: Optional[datetime] = None
    training_duration: Optional[float] = None
    
    # Deployment information
    deployment_status: Optional[str] = None
    prediction_count: Optional[int] = 0
    last_prediction_time: Optional[datetime] = None
    
    # Error handling
    error_message: Optional[str] = None
    
    class Settings:
        name = "ml_models"
        indexes = [
            "model_id",
            "dataset_id",
            "task_type",
            "status",
            "created_at"
        ]


class ReportDocument(Document):
    """Report document for MongoDB."""
    
    report_id: Indexed(str, unique=True)
    title: str
    description: Optional[str] = None
    report_type: ReportType
    
    # Report content
    datasets: List[str]  # List of dataset IDs
    sections: Optional[List[str]] = None
    
    # Generated files
    html_path: Optional[str] = None
    pdf_path: Optional[str] = None
    json_data_path: Optional[str] = None
    
    # Report metadata
    total_pages: Optional[int] = None
    word_count: Optional[int] = None
    chart_count: Optional[int] = None
    table_count: Optional[int] = None
    
    # Generation information
    status: str = "pending"  # pending, generating, completed, failed
    generation_time: Optional[float] = None
    file_size_bytes: Optional[int] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    generated_at: Optional[datetime] = None
    
    # User information
    generated_by: Optional[str] = None
    
    # Access information
    download_count: Optional[int] = 0
    last_accessed: Optional[datetime] = None
    
    class Settings:
        name = "reports"
        indexes = [
            "report_id",
            "report_type",
            "created_at",
            "status"
        ]


class ChatSessionDocument(Document):
    """Chat session document for MongoDB."""
    
    session_id: Indexed(str, unique=True)
    
    # Session configuration
    active_dataset_id: Optional[str] = None
    user_preferences: Optional[Dict[str, Any]] = None
    
    # Messages
    message_count: int = 0
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Session analytics
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    session_duration: Optional[float] = None  # in seconds
    
    # Status
    is_active: bool = True
    
    class Settings:
        name = "chat_sessions"
        indexes = [
            "session_id",
            "created_at",
            "last_activity",
            "is_active"
        ]


class UserDocument(Document):
    """User document for MongoDB."""
    
    user_id: Indexed(str, unique=True)
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    
    # User preferences
    preferences: Optional[Dict[str, Any]] = None
    default_dataset: Optional[str] = None
    
    # Activity tracking
    last_login: Optional[datetime] = None
    total_uploads: int = 0
    total_analyses: int = 0
    total_models: int = 0
    total_reports: int = 0
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Status
    is_active: bool = True
    role: str = "analyst"  # analyst, admin, viewer
    
    class Settings:
        name = "users"
        indexes = [
            "user_id",
            "username",
            "email",
            "created_at"
        ]


class RwandaProvinceDocument(Document):
    """Rwanda Province document for MongoDB."""
    
    province_id: Indexed(str, unique=True)
    province_name: str
    province_code: str
    
    # Geographic information
    area_km2: Optional[float] = None
    population: Optional[int] = None
    population_density: Optional[float] = None
    
    # Administrative
    capital_city: Optional[str] = None
    districts_count: Optional[int] = None
    
    # Economic indicators
    gdp_contribution: Optional[float] = None
    main_economic_activities: Optional[List[str]] = None
    
    class Settings:
        name = "rwanda_provinces"


class RwandaDistrictDocument(Document):
    """Rwanda District document for MongoDB."""
    
    district_id: Indexed(str, unique=True)
    district_name: str
    district_code: str
    province_id: str
    
    # Geographic information
    area_km2: Optional[float] = None
    population: Optional[int] = None
    population_density: Optional[float] = None
    
    # Administrative
    sectors_count: Optional[int] = None
    cells_count: Optional[int] = None
    villages_count: Optional[int] = None
    
    class Settings:
        name = "rwanda_districts"
        indexes = [
            "district_id",
            "province_id",
            "district_name"
        ]
