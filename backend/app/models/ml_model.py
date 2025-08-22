"""Pydantic models for machine learning operations."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class MLTaskType(str, Enum):
    """Machine learning task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES_FORECASTING = "time_series_forecasting"


class MLAlgorithm(str, Enum):
    """Supported machine learning algorithms."""
    # Classification
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"
    GRADIENT_BOOSTING_CLASSIFIER = "gradient_boosting_classifier"
    SVM_CLASSIFIER = "svm_classifier"
    NAIVE_BAYES = "naive_bayes"
    KNN_CLASSIFIER = "knn_classifier"
    NEURAL_NETWORK_CLASSIFIER = "neural_network_classifier"
    
    # Regression
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    GRADIENT_BOOSTING_REGRESSOR = "gradient_boosting_regressor"
    SVR = "svr"
    KNN_REGRESSOR = "knn_regressor"
    NEURAL_NETWORK_REGRESSOR = "neural_network_regressor"
    
    # Clustering
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL_CLUSTERING = "hierarchical_clustering"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    
    # Anomaly Detection
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"


class ModelStatus(str, Enum):
    """Model training status."""
    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"


class CrossValidationResult(BaseModel):
    """Cross-validation results."""
    cv_folds: int
    mean_score: float
    std_score: float
    scores: List[float]
    scoring_metric: str


class ModelMetrics(BaseModel):
    """Model evaluation metrics."""
    # Common metrics
    accuracy: Optional[float] = None
    
    # Classification metrics
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Optional[Dict[str, Any]] = None
    
    # Regression metrics
    mae: Optional[float] = None  # Mean Absolute Error
    mse: Optional[float] = None  # Mean Squared Error
    rmse: Optional[float] = None  # Root Mean Squared Error
    r2_score: Optional[float] = None
    
    # Clustering metrics
    silhouette_score: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    davies_bouldin_score: Optional[float] = None
    
    # Cross-validation results
    cv_results: Optional[CrossValidationResult] = None


class FeatureImportance(BaseModel):
    """Feature importance information."""
    feature_name: str
    importance: float
    rank: int


class ModelExplanation(BaseModel):
    """Model explanation and interpretability."""
    feature_importance: List[FeatureImportance] = Field(default_factory=list)
    shap_values_available: bool = False
    lime_explanations_available: bool = False
    global_explanation: Optional[Dict[str, Any]] = None
    sample_predictions_explanation: Optional[List[Dict[str, Any]]] = None


class HyperparameterResult(BaseModel):
    """Hyperparameter optimization result."""
    best_params: Dict[str, Any]
    best_score: float
    optimization_method: str  # grid_search, random_search, bayesian
    n_trials: int
    optimization_time: float


class MLModelInfo(BaseModel):
    """Machine learning model information."""
    model_config = {"protected_namespaces": ()}
    
    model_id: str
    model_name: str
    dataset_id: str
    task_type: MLTaskType
    algorithm: MLAlgorithm
    
    # Training configuration
    target_column: str
    feature_columns: List[str]
    test_size: float = 0.2
    random_state: int = 42
    
    # Model status and metadata
    status: ModelStatus = ModelStatus.PENDING
    training_start_time: Optional[datetime] = None
    training_end_time: Optional[datetime] = None
    training_duration: Optional[float] = None  # seconds
    
    # Model performance
    metrics: Optional[ModelMetrics] = None
    explanation: Optional[ModelExplanation] = None
    hyperparameter_optimization: Optional[HyperparameterResult] = None
    
    # Model artifacts
    model_file_path: Optional[str] = None
    model_size_bytes: Optional[int] = None
    
    # Error information
    error_message: Optional[str] = None
    error_timestamp: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class MLTrainingRequest(BaseModel):
    """Request model for ML model training."""
    dataset_id: str
    model_name: str
    task_type: MLTaskType
    target_column: str
    
    # Optional training configuration
    algorithms: Optional[List[MLAlgorithm]] = None  # If None, auto-select
    feature_columns: Optional[List[str]] = None  # If None, use all except target
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    cv_folds: int = Field(default=5, ge=3, le=10)
    random_state: int = 42
    
    # AutoML options
    enable_hyperparameter_tuning: bool = True
    optimization_method: str = Field(default="bayesian", pattern="^(grid_search|random_search|bayesian)$")
    max_optimization_time: int = Field(default=300, ge=60, le=3600)  # seconds
    
    # Feature engineering options
    enable_feature_selection: bool = True
    enable_feature_engineering: bool = True
    
    # Model interpretation
    generate_explanations: bool = True
    enable_shap: bool = True
    enable_lime: bool = False


class MLTrainingResponse(BaseModel):
    """Response model for ML model training."""
    model_id: str
    model_name: str
    status: ModelStatus
    message: str
    estimated_training_time: Optional[int] = None  # seconds
    created_at: datetime


class MLModelListItem(BaseModel):
    """ML model item for list responses."""
    model_config = {"protected_namespaces": ()}
    
    model_id: str
    model_name: str
    dataset_id: str
    task_type: MLTaskType
    algorithm: MLAlgorithm
    status: ModelStatus
    best_score: Optional[float] = None
    training_duration: Optional[float] = None
    created_at: datetime


class MLModelListResponse(BaseModel):
    """Response model for ML model listing."""
    models: List[MLModelListItem]
    total_count: int
    page: int = 1
    page_size: int = 50


class MLPredictionRequest(BaseModel):
    """Request model for model predictions."""
    model_config = {"protected_namespaces": ()}
    
    model_id: str
    data: Union[Dict[str, Any], List[Dict[str, Any]]]  # Single instance or batch
    explain_predictions: bool = False


class MLPredictionResponse(BaseModel):
    """Response model for model predictions."""
    model_config = {"protected_namespaces": ()}
    
    model_id: str
    predictions: Union[Any, List[Any]]  # Single prediction or batch
    probabilities: Optional[Union[List[float], List[List[float]]]] = None  # For classification
    explanations: Optional[List[Dict[str, Any]]] = None
    prediction_time: float
    generated_at: datetime = Field(default_factory=datetime.now)


class ModelComparisonRequest(BaseModel):
    """Request model for comparing multiple models."""
    model_config = {"protected_namespaces": ()}
    
    model_ids: List[str] = Field(min_items=2, max_items=10)
    comparison_metrics: List[str] = Field(default_factory=list)


class ModelComparisonResult(BaseModel):
    """Model comparison results."""
    model_config = {"protected_namespaces": ()}
    
    models: List[MLModelListItem]
    comparison_table: Dict[str, Dict[str, Any]]  # model_id -> metrics
    best_model_id: str
    best_model_metric: str
    best_model_score: float
    comparison_generated_at: datetime = Field(default_factory=datetime.now)


class AutoMLRequest(BaseModel):
    """Request model for AutoML training."""
    model_config = {"protected_namespaces": ()}
    
    dataset_id: str
    target_column: str
    task_type: Optional[MLTaskType] = None  # Auto-detect if None
    model_name_prefix: str = "AutoML"
    
    # AutoML configuration
    max_training_time: int = Field(default=1800, ge=300, le=7200)  # seconds
    max_models: int = Field(default=10, ge=5, le=50)
    enable_ensemble: bool = True
    enable_stacking: bool = True
    
    # Data preprocessing
    auto_preprocessing: bool = True
    auto_feature_engineering: bool = True
    auto_feature_selection: bool = True


class AutoMLResponse(BaseModel):
    """Response model for AutoML training."""
    session_id: str
    status: str
    message: str
    estimated_completion_time: Optional[datetime] = None
    models_to_train: List[MLAlgorithm]
    started_at: datetime = Field(default_factory=datetime.now)
