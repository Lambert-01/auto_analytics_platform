"""Pydantic models for report generation operations."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ReportType(str, Enum):
    """Types of reports that can be generated."""
    DATA_PROFILE = "data_profile"
    ANALYSIS_SUMMARY = "analysis_summary"
    ML_MODEL_REPORT = "ml_model_report"
    COMPLETE_ANALYTICS = "complete_analytics"
    EXECUTIVE_SUMMARY = "executive_summary"
    CUSTOM = "custom"


class ReportFormat(str, Enum):
    """Report output formats."""
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    MARKDOWN = "markdown"


class ReportStatus(str, Enum):
    """Report generation status."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class ChartConfig(BaseModel):
    """Configuration for charts in reports."""
    chart_type: str  # bar, line, scatter, histogram, box, heatmap, etc.
    title: str
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    width: int = 800
    height: int = 600
    interactive: bool = True


class ReportSection(BaseModel):
    """A section within a report."""
    section_id: str
    title: str
    content_type: str  # text, table, chart, statistics, insights
    content: Dict[str, Any]
    order: int = 0
    include_in_summary: bool = True


class ReportTemplate(BaseModel):
    """Report template configuration."""
    template_id: str
    name: str
    description: str
    report_type: ReportType
    sections: List[ReportSection] = Field(default_factory=list)
    default_format: ReportFormat = ReportFormat.HTML
    is_default: bool = False
    created_at: datetime = Field(default_factory=datetime.now)


class ReportMetadata(BaseModel):
    """Report metadata and information."""
    model_config = {"protected_namespaces": ()}
    
    report_id: str
    title: str
    description: Optional[str] = None
    report_type: ReportType
    format: ReportFormat
    
    # Source information
    dataset_id: Optional[str] = None
    analysis_id: Optional[str] = None
    model_id: Optional[str] = None
    
    # Generation information
    status: ReportStatus = ReportStatus.PENDING
    generation_start_time: Optional[datetime] = None
    generation_end_time: Optional[datetime] = None
    generation_duration: Optional[float] = None  # seconds
    
    # File information
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    
    # Template information
    template_id: Optional[str] = None
    custom_template: bool = False
    
    # Error information
    error_message: Optional[str] = None
    error_timestamp: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ReportGenerationRequest(BaseModel):
    """Request model for report generation."""
    model_config = {"protected_namespaces": ()}
    
    title: str
    report_type: ReportType
    format: ReportFormat = ReportFormat.HTML
    
    # Source data
    dataset_id: Optional[str] = None
    analysis_id: Optional[str] = None
    model_id: Optional[str] = None
    
    # Template and customization
    template_id: Optional[str] = None
    custom_sections: List[ReportSection] = Field(default_factory=list)
    
    # Report options
    include_raw_data: bool = False
    include_data_sample: bool = True
    sample_size: int = 100
    include_charts: bool = True
    include_statistics: bool = True
    include_insights: bool = True
    include_recommendations: bool = True
    
    # Chart options
    chart_style: str = "modern"  # modern, classic, minimal
    chart_color_scheme: str = "default"  # default, viridis, plasma, etc.
    
    # Export options
    include_interactive_elements: bool = True
    compress_output: bool = False


class ReportGenerationResponse(BaseModel):
    """Response model for report generation."""
    report_id: str
    title: str
    status: ReportStatus
    message: str
    estimated_generation_time: Optional[int] = None  # seconds
    created_at: datetime


class ReportListItem(BaseModel):
    """Report item for list responses."""
    report_id: str
    title: str
    report_type: ReportType
    format: ReportFormat
    status: ReportStatus
    file_size: Optional[int] = None
    generation_duration: Optional[float] = None
    created_at: datetime


class ReportListResponse(BaseModel):
    """Response model for report listing."""
    reports: List[ReportListItem]
    total_count: int
    page: int = 1
    page_size: int = 50


class ReportDownloadResponse(BaseModel):
    """Response model for report download."""
    report_id: str
    filename: str
    file_path: str
    file_size: int
    content_type: str
    generated_at: datetime


class ExecutiveSummary(BaseModel):
    """Executive summary for reports."""
    dataset_overview: str
    key_findings: List[str]
    data_quality_assessment: str
    recommendations: List[str]
    risk_factors: List[str] = Field(default_factory=list)
    business_impact: Optional[str] = None


class DataProfileSummary(BaseModel):
    """Data profiling summary for reports."""
    total_rows: int
    total_columns: int
    data_types_summary: Dict[str, int]
    missing_data_summary: Dict[str, float]
    data_quality_score: float
    top_correlations: List[Dict[str, Any]] = Field(default_factory=list)
    outlier_summary: Dict[str, int]
    data_distribution_insights: List[str] = Field(default_factory=list)


class ModelPerformanceSummary(BaseModel):
    """ML model performance summary for reports."""
    model_config = {"protected_namespaces": ()}
    
    model_name: str
    algorithm: str
    task_type: str
    primary_metric: str
    primary_score: float
    cross_validation_score: float
    feature_importance_top5: List[Dict[str, Any]]
    model_strengths: List[str]
    model_limitations: List[str]
    deployment_readiness: str  # ready, needs_improvement, not_ready


class ReportContent(BaseModel):
    """Complete report content."""
    model_config = {"protected_namespaces": ()}
    
    metadata: ReportMetadata
    executive_summary: Optional[ExecutiveSummary] = None
    data_profile: Optional[DataProfileSummary] = None
    model_performance: Optional[ModelPerformanceSummary] = None
    sections: List[ReportSection] = Field(default_factory=list)
    charts: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class CustomReportRequest(BaseModel):
    """Request model for custom report generation."""
    model_config = {"protected_namespaces": ()}
    
    title: str
    description: Optional[str] = None
    format: ReportFormat = ReportFormat.HTML
    
    # Data sources
    dataset_ids: List[str] = Field(default_factory=list)
    analysis_ids: List[str] = Field(default_factory=list)
    model_ids: List[str] = Field(default_factory=list)
    
    # Custom sections
    sections: List[ReportSection]
    
    # Layout options
    layout: str = "default"  # default, two_column, dashboard
    theme: str = "modern"  # modern, classic, minimal
    
    # Content options
    include_table_of_contents: bool = True
    include_appendix: bool = False
    include_methodology: bool = True


class ReportTemplateRequest(BaseModel):
    """Request model for creating report templates."""
    name: str
    description: str
    report_type: ReportType
    sections: List[ReportSection]
    default_format: ReportFormat = ReportFormat.HTML
    is_public: bool = False


class ReportTemplateResponse(BaseModel):
    """Response model for report template operations."""
    template_id: str
    name: str
    description: str
    report_type: ReportType
    sections_count: int
    is_public: bool
    created_at: datetime


class ReportAnalytics(BaseModel):
    """Analytics for report usage and performance."""
    total_reports_generated: int
    reports_by_type: Dict[ReportType, int]
    reports_by_format: Dict[ReportFormat, int]
    average_generation_time: float
    most_popular_templates: List[Dict[str, Any]]
    recent_activity: List[Dict[str, Any]]
    generation_date_range: tuple[datetime, datetime]
