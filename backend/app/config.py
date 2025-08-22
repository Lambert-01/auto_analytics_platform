"""Configuration settings for the Auto Analytics Platform."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings configuration."""
    
    # API Configuration
    app_name: str = Field(default="NISR Rwanda Analytics Platform", env="APP_NAME")
    api_title: str = "NISR Analytics API"
    api_description: str = "Rwanda National Statistics Analytics Platform"
    api_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="production", env="ENVIRONMENT")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./nisr_analytics.db",
        env="DATABASE_URL"
    )
    database_host: Optional[str] = Field(default=None, env="DATABASE_HOST")
    database_port: Optional[int] = Field(default=None, env="DATABASE_PORT")
    database_name: Optional[str] = Field(default=None, env="DATABASE_NAME")
    database_user: Optional[str] = Field(default=None, env="DATABASE_USER")
    database_password: Optional[str] = Field(default=None, env="DATABASE_PASSWORD")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    # Redis Configuration (for Celery)
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # File Storage Configuration
    upload_dir: Path = Field(default=Path("data/uploads"))
    processed_dir: Path = Field(default=Path("data/processed"))
    models_dir: Path = Field(default=Path("data/models"))
    cache_dir: Path = Field(default=Path("data/cache"))
    reports_dir: Path = Field(default=Path("reports"))
    
    # File Upload Limits
    max_file_size: int = Field(default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    allowed_extensions: list[str] = Field(
        default=[".csv", ".xlsx", ".xls", ".json", ".parquet"]
    )
    
    # ML Configuration
    default_test_size: float = Field(default=0.2, env="DEFAULT_TEST_SIZE")
    max_training_time: int = Field(default=300, env="MAX_TRAINING_TIME")  # seconds
    cross_validation_folds: int = Field(default=5, env="CV_FOLDS")
    
    # Security Configuration
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="TOKEN_EXPIRE_MINUTES")
    
    # CORS Configuration
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/auto_analytics.log", env="LOG_FILE")
    
    # Visualization Configuration
    default_chart_width: int = Field(default=800, env="CHART_WIDTH")
    default_chart_height: int = Field(default=600, env="CHART_HEIGHT")
    chart_dpi: int = Field(default=300, env="CHART_DPI")
    
    # Report Configuration
    report_template_dir: Path = Field(default=Path("reports/templates"))
    max_report_size: int = Field(default=50 * 1024 * 1024, env="MAX_REPORT_SIZE")  # 50MB
    
    # ===== NISR Specific Configuration =====
    # External APIs
    bnr_api_key: Optional[str] = Field(default=None, env="BNR_API_KEY")
    bnr_api_url: str = Field(default="https://www.bnr.rw/statistics/", env="BNR_API_URL")
    world_bank_api_url: str = Field(default="https://api.worldbank.org/v2/", env="WORLD_BANK_API_URL")
    un_data_api_key: Optional[str] = Field(default=None, env="UN_DATA_API_KEY")
    
    # Module enablement
    enable_census_module: bool = Field(default=True, env="ENABLE_CENSUS_MODULE")
    enable_economic_module: bool = Field(default=True, env="ENABLE_ECONOMIC_MODULE")
    enable_health_module: bool = Field(default=True, env="ENABLE_HEALTH_MODULE")
    enable_education_module: bool = Field(default=True, env="ENABLE_EDUCATION_MODULE")
    enable_agriculture_module: bool = Field(default=True, env="ENABLE_AGRICULTURE_MODULE")
    
    # Geographic validation
    validate_geographic_codes: bool = Field(default=True, env="VALIDATE_GEOGRAPHIC_CODES")
    rwanda_provinces: list[str] = Field(
        default=["Kigali", "Eastern", "Northern", "Southern", "Western"]
    )
    rwanda_districts_count: int = Field(default=30, env="RWANDA_DISTRICTS_COUNT")
    
    # AI/ML Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=4096, env="OPENAI_MAX_TOKENS")
    automl_time_limit: int = Field(default=3600, env="AUTOML_TIME_LIMIT")
    automl_max_models: int = Field(default=50, env="AUTOML_MAX_MODELS")
    enable_gpu: bool = Field(default=False, env="ENABLE_GPU")
    
    # Feature flags
    enable_real_time_monitoring: bool = Field(default=True, env="ENABLE_REAL_TIME_MONITORING")
    enable_advanced_visualizations: bool = Field(default=True, env="ENABLE_ADVANCED_VISUALIZATIONS")
    enable_automated_reports: bool = Field(default=True, env="ENABLE_AUTOMATED_REPORTS")
    enable_ml_predictions: bool = Field(default=True, env="ENABLE_ML_PREDICTIONS")
    enable_data_quality_checks: bool = Field(default=True, env="ENABLE_DATA_QUALITY_CHECKS")
    
    # Performance settings
    worker_memory_limit: int = Field(default=2048, env="WORKER_MEMORY_LIMIT")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    max_concurrent_jobs: int = Field(default=5, env="MAX_CONCURRENT_JOBS")
    
    # Compliance
    data_retention_days: int = Field(default=2555, env="DATA_RETENTION_DAYS")  # 7 years
    enable_data_anonymization: bool = Field(default=True, env="ENABLE_DATA_ANONYMIZATION")
    gdpr_compliance: bool = Field(default=True, env="GDPR_COMPLIANCE")
    pii_detection: bool = Field(default=True, env="PII_DETECTION")
    
    # Localization
    timezone: str = Field(default="Africa/Kigali", env="TIMEZONE")
    currency: str = Field(default="RWF", env="CURRENCY")
    language: str = Field(default="en", env="LANGUAGE")
    locale: str = Field(default="rw_RW", env="LOCALE")
    
    # Administrative defaults
    default_province: str = Field(default="Kigali", env="DEFAULT_PROVINCE")
    default_district: str = Field(default="Gasabo", env="DEFAULT_DISTRICT")
    default_sector: str = Field(default="Kimisagara", env="DEFAULT_SECTOR")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **kwargs):
        """Initialize settings and create necessary directories."""
        super().__init__(**kwargs)
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.upload_dir,
            self.processed_dir,
            self.models_dir,
            self.cache_dir,
            self.reports_dir,
            self.report_template_dir,
            Path("logs")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def database_url_postgres(self) -> str:
        """Construct PostgreSQL database URL from components."""
        if all([self.database_user, self.database_password, self.database_host, self.database_name]):
            return f"postgresql://{self.database_user}:{self.database_password}@{self.database_host}:{self.database_port or 5432}/{self.database_name}"
        return self.database_url
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() in ["development", "dev"]


# Global settings instance
settings = Settings()
