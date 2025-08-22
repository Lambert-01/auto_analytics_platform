"""Pydantic models for data analysis operations."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class AnalysisType(str, Enum):
    """Types of analysis that can be performed."""
    DESCRIPTIVE = "descriptive"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    OUTLIER_DETECTION = "outlier_detection"
    FEATURE_IMPORTANCE = "feature_importance"
    CLUSTERING = "clustering"
    ASSOCIATION_RULES = "association_rules"
    TREND_ANALYSIS = "trend_analysis"


class StatisticalSummary(BaseModel):
    """Statistical summary for numerical columns."""
    count: int
    mean: float
    std: float
    min: float
    q25: float
    q50: float  # median
    q75: float
    max: float
    skewness: float
    kurtosis: float
    variance: float


class CategoricalSummary(BaseModel):
    """Summary for categorical columns."""
    count: int
    unique: int
    top: str  # most frequent value
    freq: int  # frequency of most frequent value
    mode: str
    entropy: float
    value_counts: Dict[str, int]


class ColumnAnalysis(BaseModel):
    """Analysis results for a single column."""
    column_name: str
    data_type: str
    is_numerical: bool
    is_categorical: bool
    is_datetime: bool
    
    # Statistical summaries
    numerical_summary: Optional[StatisticalSummary] = None
    categorical_summary: Optional[CategoricalSummary] = None
    
    # Quality metrics
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    
    # Outlier information
    outlier_count: int = 0
    outlier_percentage: float = 0.0
    outlier_method: Optional[str] = None


class CorrelationAnalysis(BaseModel):
    """Correlation analysis results."""
    method: str = "pearson"  # pearson, spearman, kendall
    correlation_matrix: Dict[str, Dict[str, float]]
    strong_correlations: List[Dict[str, Any]] = Field(default_factory=list)
    correlation_threshold: float = 0.7


class OutlierDetectionResult(BaseModel):
    """Outlier detection results."""
    method: str  # iqr, zscore, isolation_forest
    total_outliers: int
    outlier_percentage: float
    outliers_by_column: Dict[str, List[int]]  # column -> list of row indices
    outlier_summary: Dict[str, int]  # column -> count


class ClusteringResult(BaseModel):
    """Clustering analysis results."""
    algorithm: str  # kmeans, dbscan, hierarchical
    n_clusters: int
    cluster_labels: List[int]
    cluster_centers: Optional[List[List[float]]] = None
    silhouette_score: Optional[float] = None
    inertia: Optional[float] = None
    cluster_summary: Dict[int, Dict[str, Any]]  # cluster_id -> summary stats


class FeatureImportanceResult(BaseModel):
    """Feature importance analysis results."""
    method: str  # random_forest, xgboost, mutual_info
    target_column: str
    importance_scores: Dict[str, float]  # feature -> importance score
    ranked_features: List[str]  # features sorted by importance
    top_n_features: List[str] = Field(default_factory=list)


class TrendAnalysisResult(BaseModel):
    """Trend analysis results for time series data."""
    date_column: str
    value_columns: List[str]
    trends: Dict[str, str]  # column -> trend direction (increasing, decreasing, stable)
    seasonal_patterns: Dict[str, bool]  # column -> has_seasonal_pattern
    trend_strength: Dict[str, float]  # column -> trend strength (0-1)
    changepoints: Dict[str, List[datetime]]  # column -> list of changepoint dates


class DataAnalysisRequest(BaseModel):
    """Request model for data analysis."""
    dataset_id: str
    analysis_types: List[AnalysisType]
    options: Dict[str, Any] = Field(default_factory=dict)
    
    # Specific analysis options
    correlation_threshold: float = 0.7
    outlier_method: str = "iqr"
    clustering_algorithm: str = "kmeans"
    n_clusters: Optional[int] = None
    target_column: Optional[str] = None  # for supervised analysis
    date_column: Optional[str] = None  # for time series analysis


class DataAnalysisResult(BaseModel):
    """Complete data analysis results."""
    dataset_id: str
    analysis_id: str
    dataset_shape: tuple[int, int]
    
    # Column-wise analysis
    column_analyses: List[ColumnAnalysis] = Field(default_factory=list)
    
    # Cross-column analyses
    correlation_analysis: Optional[CorrelationAnalysis] = None
    outlier_detection: Optional[OutlierDetectionResult] = None
    clustering_result: Optional[ClusteringResult] = None
    feature_importance: Optional[FeatureImportanceResult] = None
    trend_analysis: Optional[TrendAnalysisResult] = None
    
    # Overall insights
    key_insights: List[str] = Field(default_factory=list)
    data_quality_score: float = Field(ge=0, le=100)
    recommendations: List[str] = Field(default_factory=list)
    
    # Metadata
    analysis_types_performed: List[AnalysisType]
    execution_time: float
    generated_at: datetime = Field(default_factory=datetime.now)


class AnalysisListItem(BaseModel):
    """Analysis item for list responses."""
    analysis_id: str
    dataset_id: str
    dataset_filename: str
    analysis_types: List[AnalysisType]
    data_quality_score: float
    execution_time: float
    generated_at: datetime


class AnalysisListResponse(BaseModel):
    """Response model for analysis listing."""
    analyses: List[AnalysisListItem]
    total_count: int
    page: int = 1
    page_size: int = 50


class QuickInsight(BaseModel):
    """Quick insight about the dataset."""
    type: str  # insight type
    title: str
    description: str
    importance: str = "medium"  # low, medium, high
    affected_columns: List[str] = Field(default_factory=list)
    suggested_action: Optional[str] = None


class QuickAnalysisResponse(BaseModel):
    """Quick analysis response with key insights."""
    dataset_id: str
    quick_insights: List[QuickInsight]
    summary_stats: Dict[str, Any]
    data_quality_score: float
    execution_time: float
    generated_at: datetime = Field(default_factory=datetime.now)
