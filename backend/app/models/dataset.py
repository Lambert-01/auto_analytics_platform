"""Pydantic models for dataset operations."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class DatasetStatus(str, Enum):
    """Dataset processing status."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    ERROR = "error"


class DatasetType(str, Enum):
    """Dataset type classification."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    MIXED = "mixed"
    TEXT = "text"
    TIME_SERIES = "time_series"


class ColumnInfo(BaseModel):
    """Information about a dataset column."""
    name: str
    data_type: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    sample_values: List[Any] = Field(default_factory=list)
    statistics: Optional[Dict[str, Any]] = None


class DatasetMetadata(BaseModel):
    """Dataset metadata and basic information."""
    file_id: str
    filename: str
    original_filename: str
    file_size: int
    file_hash: str
    upload_timestamp: datetime
    
    # Data characteristics
    rows: int
    columns: int
    dataset_type: Optional[DatasetType] = None
    missing_values_count: int = 0
    missing_values_percentage: float = 0.0
    
    # Column information
    column_info: List[ColumnInfo] = Field(default_factory=list)
    
    # Processing status
    status: DatasetStatus = DatasetStatus.UPLOADED
    processing_start_time: Optional[datetime] = None
    processing_end_time: Optional[datetime] = None
    processing_duration: Optional[float] = None  # seconds
    
    # Error information
    error_message: Optional[str] = None
    error_timestamp: Optional[datetime] = None


class DatasetUploadRequest(BaseModel):
    """Request model for dataset upload."""
    filename: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class DatasetUploadResponse(BaseModel):
    """Response model for dataset upload."""
    file_id: str
    filename: str
    file_size: int
    status: DatasetStatus
    message: str
    upload_timestamp: datetime


class DatasetListItem(BaseModel):
    """Dataset item for list responses."""
    file_id: str
    filename: str
    original_filename: str
    file_size: int
    rows: int
    columns: int
    dataset_type: Optional[DatasetType]
    status: DatasetStatus
    upload_timestamp: datetime
    processing_duration: Optional[float]


class DatasetListResponse(BaseModel):
    """Response model for dataset listing."""
    datasets: List[DatasetListItem]
    total_count: int
    page: int = 1
    page_size: int = 50


class DatasetDetailResponse(BaseModel):
    """Detailed dataset information response."""
    metadata: DatasetMetadata
    sample_data: List[Dict[str, Any]] = Field(default_factory=list)
    basic_statistics: Optional[Dict[str, Any]] = None


class DatasetDeleteResponse(BaseModel):
    """Response model for dataset deletion."""
    file_id: str
    filename: str
    deleted: bool
    message: str
    timestamp: datetime


class DataQualityIssue(BaseModel):
    """Data quality issue information."""
    type: str  # 'missing_values', 'duplicates', 'outliers', 'inconsistent_format'
    column: Optional[str] = None
    count: int
    percentage: float
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    suggested_action: Optional[str] = None


class DataQualityReport(BaseModel):
    """Data quality assessment report."""
    dataset_id: str
    overall_score: float = Field(ge=0, le=100)  # 0-100 quality score
    issues: List[DataQualityIssue] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)


class DataPreprocessingOptions(BaseModel):
    """Options for data preprocessing."""
    handle_missing_values: bool = True
    missing_value_strategy: str = Field(default="auto", pattern="^(auto|drop|mean|median|mode|forward_fill|backward_fill)$")
    remove_duplicates: bool = True
    handle_outliers: bool = True
    outlier_method: str = Field(default="iqr", pattern="^(iqr|zscore|isolation_forest)$")
    encode_categorical: bool = True
    encoding_method: str = Field(default="auto", pattern="^(auto|onehot|label|target)$")
    scale_numerical: bool = True
    scaling_method: str = Field(default="standard", pattern="^(standard|minmax|robust|quantile)$")


class DataPreprocessingResult(BaseModel):
    """Result of data preprocessing."""
    dataset_id: str
    processed_file_path: str
    original_shape: tuple[int, int]
    processed_shape: tuple[int, int]
    preprocessing_steps: List[str]
    execution_time: float
    quality_improvement: float  # Quality score improvement
    generated_at: datetime = Field(default_factory=datetime.now)
