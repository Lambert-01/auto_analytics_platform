"""Configuration settings for the Auto Analytics Platform."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings configuration."""
    
    # API Configuration
    api_title: str = "Auto Analytics Platform API"
    api_description: str = "Automated data analysis, ML, and report generation"
    api_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./auto_analytics.db",
        env="DATABASE_URL"
    )
    
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


# Global settings instance
settings = Settings()
