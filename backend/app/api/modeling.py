"""Machine learning modeling API endpoints."""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.models.ml_model import (
    MLTrainingRequest,
    MLTrainingResponse,
    MLModelListResponse,
    MLModelListItem,
    MLModelInfo,
    MLPredictionRequest,
    MLPredictionResponse,
    ModelComparisonRequest,
    ModelComparisonResult,
    AutoMLRequest,
    AutoMLResponse,
    ModelStatus,
    MLTaskType,
    MLAlgorithm
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/models/train", response_model=MLTrainingResponse)
async def train_model(
    request: MLTrainingRequest,
    background_tasks: BackgroundTasks
):
    """Train a machine learning model.
    
    Args:
        request: Model training configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Training job information
    """
    try:
        logger.info(f"Starting model training for dataset: {request.dataset_id}")
        
        # Generate model ID
        model_id = f"model_{request.dataset_id}_{int(datetime.now().timestamp())}"
        
        # Start background training task
        background_tasks.add_task(
            train_ml_model,
            model_id,
            request
        )
        
        return MLTrainingResponse(
            model_id=model_id,
            model_name=request.model_name,
            status=ModelStatus.PENDING,
            message="Model training started successfully",
            estimated_training_time=request.max_optimization_time if request.enable_hyperparameter_tuning else 180,
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(status_code=500, detail="Error starting model training")


@router.get("/models", response_model=MLModelListResponse)
async def list_models(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    dataset_id: Optional[str] = Query(None, description="Filter by dataset"),
    task_type: Optional[MLTaskType] = Query(None, description="Filter by task type"),
    status: Optional[ModelStatus] = Query(None, description="Filter by status")
):
    """List all machine learning models with pagination.
    
    Args:
        page: Page number
        page_size: Items per page
        dataset_id: Filter by dataset ID
        task_type: Filter by ML task type
        status: Filter by model status
        
    Returns:
        Paginated list of ML models
    """
    try:
        # TODO: Implement database queries
        # For now, return sample data
        
        sample_models = [
            MLModelListItem(
                model_id="model_1",
                model_name="Sales Prediction Model",
                dataset_id="dataset_1",
                task_type=MLTaskType.REGRESSION,
                algorithm=MLAlgorithm.RANDOM_FOREST_REGRESSOR,
                status=ModelStatus.COMPLETED,
                best_score=0.87,
                training_duration=245.6,
                created_at=datetime.now()
            ),
            MLModelListItem(
                model_id="model_2",
                model_name="Customer Segmentation",
                dataset_id="dataset_2",
                task_type=MLTaskType.CLUSTERING,
                algorithm=MLAlgorithm.KMEANS,
                status=ModelStatus.COMPLETED,
                best_score=0.72,
                training_duration=156.3,
                created_at=datetime.now()
            )
        ]
        
        # Apply filters
        filtered_models = sample_models
        if dataset_id:
            filtered_models = [m for m in filtered_models if m.dataset_id == dataset_id]
        if task_type:
            filtered_models = [m for m in filtered_models if m.task_type == task_type]
        if status:
            filtered_models = [m for m in filtered_models if m.status == status]
        
        return MLModelListResponse(
            models=filtered_models,
            total_count=len(filtered_models),
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving models")


@router.get("/models/{model_id}", response_model=MLModelInfo)
async def get_model_details(model_id: str):
    """Get detailed information about a specific model.
    
    Args:
        model_id: Unique model identifier
        
    Returns:
        Detailed model information
    """
    try:
        # TODO: Implement database query for model details
        # For now, return sample data
        
        from app.models.ml_model import ModelMetrics, ModelExplanation, FeatureImportance
        
        sample_metrics = ModelMetrics(
            accuracy=0.87,
            precision=0.85,
            recall=0.89,
            f1_score=0.87,
            roc_auc=0.92,
            r2_score=0.87,
            mae=15.2,
            mse=342.7,
            rmse=18.5
        )
        
        sample_explanation = ModelExplanation(
            feature_importance=[
                FeatureImportance(feature_name="price", importance=0.45, rank=1),
                FeatureImportance(feature_name="quantity", importance=0.32, rank=2),
                FeatureImportance(feature_name="discount", importance=0.23, rank=3)
            ],
            shap_values_available=True,
            lime_explanations_available=False
        )
        
        return MLModelInfo(
            model_id=model_id,
            model_name="Sales Prediction Model",
            dataset_id="dataset_1",
            task_type=MLTaskType.REGRESSION,
            algorithm=MLAlgorithm.RANDOM_FOREST_REGRESSOR,
            target_column="sales_amount",
            feature_columns=["price", "quantity", "discount", "category"],
            status=ModelStatus.COMPLETED,
            training_start_time=datetime.now(),
            training_end_time=datetime.now(),
            training_duration=245.6,
            metrics=sample_metrics,
            explanation=sample_explanation,
            model_file_path=f"data/models/{model_id}.joblib",
            model_size_bytes=2048000
        )
        
    except Exception as e:
        logger.error(f"Error getting model details for {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving model details")


@router.post("/models/{model_id}/predict", response_model=MLPredictionResponse)
async def make_prediction(model_id: str, request: MLPredictionRequest):
    """Make predictions using a trained model.
    
    Args:
        model_id: Unique model identifier
        request: Prediction request data
        
    Returns:
        Prediction results
    """
    try:
        logger.info(f"Making prediction with model: {model_id}")
        
        # TODO: Implement actual prediction logic
        # This would include:
        # - Loading the trained model
        # - Preprocessing input data
        # - Making predictions
        # - Generating explanations if requested
        
        # For now, return sample predictions
        import random
        
        if isinstance(request.data, list):
            # Batch prediction
            predictions = [random.uniform(50, 200) for _ in request.data]
            probabilities = [[random.uniform(0, 1) for _ in range(3)] for _ in request.data] if request.explain_predictions else None
        else:
            # Single prediction
            predictions = random.uniform(50, 200)
            probabilities = [random.uniform(0, 1) for _ in range(3)] if request.explain_predictions else None
        
        explanations = None
        if request.explain_predictions:
            explanations = [
                {
                    "feature": "price",
                    "contribution": 0.45,
                    "value": 100.0
                },
                {
                    "feature": "quantity", 
                    "contribution": 0.32,
                    "value": 5
                }
            ]
        
        return MLPredictionResponse(
            model_id=model_id,
            predictions=predictions,
            probabilities=probabilities,
            explanations=explanations,
            prediction_time=0.15
        )
        
    except Exception as e:
        logger.error(f"Error making prediction with model {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Error making prediction")


@router.get("/models/{model_id}/status")
async def get_model_status(model_id: str):
    """Get the current status of model training.
    
    Args:
        model_id: Unique model identifier
        
    Returns:
        Current training status and progress
    """
    try:
        # TODO: Implement actual status tracking
        
        return {
            "model_id": model_id,
            "status": "completed",
            "progress": 100,
            "current_step": "Training complete",
            "steps_completed": [
                "Data preprocessing",
                "Feature selection",
                "Model training",
                "Hyperparameter optimization",
                "Model evaluation",
                "Model saving"
            ],
            "training_time": 245.6,
            "estimated_remaining_time": 0,
            "best_score": 0.87,
            "current_algorithm": "Random Forest",
            "updated_at": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting model status for {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving model status")


@router.post("/models/compare", response_model=ModelComparisonResult)
async def compare_models(request: ModelComparisonRequest):
    """Compare multiple models.
    
    Args:
        request: Model comparison configuration
        
    Returns:
        Model comparison results
    """
    try:
        logger.info(f"Comparing models: {request.model_ids}")
        
        # TODO: Implement actual model comparison logic
        
        sample_models = [
            MLModelListItem(
                model_id=model_id,
                model_name=f"Model {i+1}",
                dataset_id="dataset_1",
                task_type=MLTaskType.REGRESSION,
                algorithm=MLAlgorithm.RANDOM_FOREST_REGRESSOR,
                status=ModelStatus.COMPLETED,
                best_score=0.87 - i*0.05,
                training_duration=245.6,
                created_at=datetime.now()
            )
            for i, model_id in enumerate(request.model_ids)
        ]
        
        comparison_table = {
            model_id: {
                "accuracy": 0.87 - i*0.05,
                "precision": 0.85 - i*0.04,
                "recall": 0.89 - i*0.03,
                "f1_score": 0.87 - i*0.04,
                "training_time": 245.6 + i*50
            }
            for i, model_id in enumerate(request.model_ids)
        }
        
        return ModelComparisonResult(
            models=sample_models,
            comparison_table=comparison_table,
            best_model_id=request.model_ids[0],
            best_model_metric="accuracy",
            best_model_score=0.87
        )
        
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail="Error comparing models")


@router.post("/models/automl", response_model=AutoMLResponse)
async def start_automl(
    request: AutoMLRequest,
    background_tasks: BackgroundTasks
):
    """Start AutoML training session.
    
    Args:
        request: AutoML configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        AutoML session information
    """
    try:
        logger.info(f"Starting AutoML for dataset: {request.dataset_id}")
        
        # Generate session ID
        session_id = f"automl_{request.dataset_id}_{int(datetime.now().timestamp())}"
        
        # Start background AutoML task
        background_tasks.add_task(
            run_automl_session,
            session_id,
            request
        )
        
        return AutoMLResponse(
            session_id=session_id,
            status="started",
            message="AutoML session started successfully",
            estimated_completion_time=datetime.now(),
            models_to_train=[
                MLAlgorithm.RANDOM_FOREST_REGRESSOR,
                MLAlgorithm.GRADIENT_BOOSTING_REGRESSOR,
                MLAlgorithm.LINEAR_REGRESSION
            ]
        )
        
    except Exception as e:
        logger.error(f"Error starting AutoML: {e}")
        raise HTTPException(status_code=500, detail="Error starting AutoML session")


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model and its artifacts.
    
    Args:
        model_id: Unique model identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        logger.info(f"Deleting model: {model_id}")
        
        # TODO: Implement actual deletion logic
        # This would include:
        # - Removing model files
        # - Deleting database records
        # - Cleaning up associated predictions
        
        return {
            "model_id": model_id,
            "deleted": True,
            "message": "Model and artifacts deleted successfully",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Error deleting model")


async def train_ml_model(model_id: str, request: MLTrainingRequest):
    """Background task to train ML model.
    
    Args:
        model_id: Unique model identifier
        request: Training configuration
    """
    try:
        import time
        import json
        from pathlib import Path
        import pandas as pd
        
        start_time = time.time()
        logger.info(f"Starting comprehensive background training for model: {model_id}")
        
        # Find and load the dataset
        dataset_path = None
        data_folder = Path("data")
        
        # Search for dataset files
        for folder in ["uploads", "processed"]:
            folder_path = data_folder / folder
            if folder_path.exists():
                for file_path in folder_path.glob(f"*{request.dataset_id}*"):
                    if file_path.suffix in ['.csv', '.xlsx', '.parquet']:
                        dataset_path = file_path
                        break
        
        if not dataset_path:
            logger.error(f"Dataset not found for training: {request.dataset_id}")
            return
        
        # Load dataset
        if dataset_path.suffix == '.csv':
            df = pd.read_csv(dataset_path)
        elif dataset_path.suffix == '.xlsx':
            df = pd.read_excel(dataset_path)
        elif dataset_path.suffix == '.parquet':
            df = pd.read_parquet(dataset_path)
        else:
            logger.error(f"Unsupported file format: {dataset_path.suffix}")
            return
        
        logger.info(f"Dataset loaded for training: {df.shape}")
        
        # Initialize AutoML engine
        from app.core.automl_engine import AutoMLEngine
        automl_engine = AutoMLEngine()
        
        # Determine problem type from request or auto-detect
        if request.task_type == MLTaskType.CLASSIFICATION:
            problem_type = 'classification'
        elif request.task_type == MLTaskType.REGRESSION:
            problem_type = 'regression'
        else:
            problem_type = 'auto'  # Auto-detect
        
        # Train the model
        training_results = automl_engine.train_best_model(
            df=df,
            target_column=request.target_column,
            problem_type=problem_type,
            max_time=request.max_optimization_time if request.enable_hyperparameter_tuning else 300,
            cv_folds=request.cross_validation_folds,
            test_size=request.test_split_ratio
        )
        
        # Save model artifacts
        models_folder = data_folder / "models" / request.task_type.value
        models_folder.mkdir(parents=True, exist_ok=True)
        
        model_folder = models_folder / model_id
        model_folder.mkdir(exist_ok=True)
        
        # Save the trained model
        model_path = model_folder / "model.joblib"
        import joblib
        joblib.dump(training_results['best_model'], model_path)
        
        # Save model metadata
        execution_time = time.time() - start_time
        model_metadata = {
            'model_id': model_id,
            'model_name': request.model_name,
            'dataset_id': request.dataset_id,
            'dataset_path': str(dataset_path),
            'task_type': request.task_type.value,
            'target_column': request.target_column,
            'feature_columns': request.feature_columns,
            'algorithm': training_results.get('best_model_name', 'unknown'),
            'best_score': training_results.get('best_score', 0.0),
            'cv_scores': training_results.get('cv_scores', []),
            'feature_importance': training_results.get('feature_importance', {}),
            'metrics': training_results.get('test_metrics', {}),
            'training_config': {
                'max_optimization_time': request.max_optimization_time,
                'enable_hyperparameter_tuning': request.enable_hyperparameter_tuning,
                'cross_validation_folds': request.cross_validation_folds,
                'test_split_ratio': request.test_split_ratio,
                'random_state': request.random_state
            },
            'execution_time': execution_time,
            'model_path': str(model_path),
            'model_size_bytes': model_path.stat().st_size if model_path.exists() else 0,
            'created_at': time.time(),
            'status': 'completed'
        }
        
        # Save metadata
        metadata_path = model_folder / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(_make_model_serializable(model_metadata), f, indent=2, default=str)
        
        # Save training report
        if training_results.get('training_report'):
            report_path = model_folder / "training_report.json"
            with open(report_path, 'w') as f:
                json.dump(_make_model_serializable(training_results['training_report']), f, indent=2, default=str)
        
        # Update status
        status_data = {
            'model_id': model_id,
            'status': 'completed',
            'progress': 100,
            'execution_time': execution_time,
            'best_score': training_results.get('best_score', 0.0),
            'best_algorithm': training_results.get('best_model_name', 'Unknown'),
            'completed_at': time.time()
        }
        
        status_path = model_folder / "status.json"
        with open(status_path, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        logger.info(f"Model training completed: {model_id} in {execution_time:.2f} seconds")
        logger.info(f"Best model: {training_results.get('best_model_name', 'Unknown')} with score: {training_results.get('best_score', 0.0):.4f}")
        
    except Exception as e:
        logger.error(f"Error in comprehensive background training {model_id}: {e}")
        
        # Save error status
        try:
            error_folder = Path("data/models") / model_id
            error_folder.mkdir(parents=True, exist_ok=True)
            
            error_status = {
                'model_id': model_id,
                'status': 'error',
                'error_message': str(e),
                'error_timestamp': time.time(),
                'progress': 0
            }
            
            error_file = error_folder / "status.json"
            with open(error_file, 'w') as f:
                json.dump(error_status, f, indent=2)
        except Exception as save_error:
            logger.error(f"Failed to save error status: {save_error}")


def _make_model_serializable(obj):
    """Convert complex objects to JSON-serializable format for models."""
    import numpy as np
    import pandas as pd
    from datetime import datetime
    
    if isinstance(obj, dict):
        return {key: _make_model_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_model_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    elif hasattr(obj, 'to_dict'):
        return _make_model_serializable(obj.to_dict())
    elif hasattr(obj, '__dict__'):
        return _make_model_serializable(obj.__dict__)
    else:
        return obj


async def run_automl_session(session_id: str, request: AutoMLRequest):
    """Background task to run AutoML session.
    
    Args:
        session_id: Unique session identifier
        request: AutoML configuration
    """
    try:
        import time
        import json
        from pathlib import Path
        import pandas as pd
        
        start_time = time.time()
        logger.info(f"Starting comprehensive AutoML session: {session_id}")
        
        # Find and load the dataset
        dataset_path = None
        data_folder = Path("data")
        
        # Search for dataset files
        for folder in ["uploads", "processed"]:
            folder_path = data_folder / folder
            if folder_path.exists():
                for file_path in folder_path.glob(f"*{request.dataset_id}*"):
                    if file_path.suffix in ['.csv', '.xlsx', '.parquet']:
                        dataset_path = file_path
                        break
        
        if not dataset_path:
            logger.error(f"Dataset not found for AutoML: {request.dataset_id}")
            return
        
        # Load dataset
        if dataset_path.suffix == '.csv':
            df = pd.read_csv(dataset_path)
        elif dataset_path.suffix == '.xlsx':
            df = pd.read_excel(dataset_path)
        elif dataset_path.suffix == '.parquet':
            df = pd.read_parquet(dataset_path)
        else:
            logger.error(f"Unsupported file format: {dataset_path.suffix}")
            return
        
        logger.info(f"Dataset loaded for AutoML: {df.shape}")
        
        # Initialize AutoML engine
        from app.core.automl_engine import AutoMLEngine
        automl_engine = AutoMLEngine()
        
        # Run comprehensive AutoML
        automl_results = automl_engine.run_comprehensive_automl(
            df=df,
            target_column=request.target_column,
            max_time=request.max_time,
            max_models=request.max_models,
            include_ensemble=request.include_ensemble,
            cv_folds=5,
            test_size=0.2,
            random_state=42
        )
        
        # Save AutoML session results
        automl_folder = data_folder / "models" / "automl" / session_id
        automl_folder.mkdir(parents=True, exist_ok=True)
        
        # Save all trained models
        saved_models = []
        for i, (model_name, model_data) in enumerate(automl_results.get('models', {}).items()):
            model_path = automl_folder / f"model_{i}_{model_name.replace(' ', '_')}.joblib"
            import joblib
            joblib.dump(model_data['model'], model_path)
            
            saved_models.append({
                'model_name': model_name,
                'model_path': str(model_path),
                'score': model_data.get('score', 0.0),
                'cv_scores': model_data.get('cv_scores', []),
                'metrics': model_data.get('metrics', {}),
                'feature_importance': model_data.get('feature_importance', {}),
                'rank': i + 1
            })
        
        # Save the best ensemble model if available
        best_model_path = None
        if automl_results.get('best_model'):
            best_model_path = automl_folder / "best_model.joblib"
            joblib.dump(automl_results['best_model'], best_model_path)
        
        # Save comprehensive AutoML metadata
        execution_time = time.time() - start_time
        automl_metadata = {
            'session_id': session_id,
            'dataset_id': request.dataset_id,
            'dataset_path': str(dataset_path),
            'target_column': request.target_column,
            'session_config': {
                'max_time': request.max_time,
                'max_models': request.max_models,
                'include_ensemble': request.include_ensemble,
                'optimization_metric': request.optimization_metric
            },
            'results_summary': {
                'total_models_trained': len(automl_results.get('models', {})),
                'best_model_name': automl_results.get('best_model_name', 'Unknown'),
                'best_score': automl_results.get('best_score', 0.0),
                'model_comparison': automl_results.get('model_comparison', {}),
                'feature_importance_global': automl_results.get('feature_importance', {}),
            },
            'trained_models': saved_models,
            'best_model_path': str(best_model_path) if best_model_path else None,
            'preprocessing_pipeline': automl_results.get('preprocessing_info', {}),
            'data_analysis': automl_results.get('data_analysis', {}),
            'recommendations': automl_results.get('recommendations', []),
            'execution_time': execution_time,
            'completed_at': time.time(),
            'status': 'completed'
        }
        
        # Save AutoML session metadata
        metadata_path = automl_folder / "automl_session.json"
        with open(metadata_path, 'w') as f:
            json.dump(_make_model_serializable(automl_metadata), f, indent=2, default=str)
        
        # Generate AutoML report
        from app.core.report_generator import ComprehensiveReportGenerator
        report_generator = ComprehensiveReportGenerator()
        
        automl_report = report_generator.generate_automl_report(
            automl_results=automl_results,
            session_metadata=automl_metadata
        )
        
        # Save AutoML report
        if automl_report.get('html_content'):
            report_path = automl_folder / "automl_report.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(automl_report['html_content'])
            logger.info(f"AutoML HTML report saved: {report_path}")
        
        # Update session status
        status_data = {
            'session_id': session_id,
            'status': 'completed',
            'progress': 100,
            'execution_time': execution_time,
            'best_score': automl_results.get('best_score', 0.0),
            'best_model': automl_results.get('best_model_name', 'Unknown'),
            'models_trained': len(automl_results.get('models', {})),
            'completed_at': time.time(),
            'results_available': True
        }
        
        status_path = automl_folder / "status.json"
        with open(status_path, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        logger.info(f"Comprehensive AutoML session completed: {session_id} in {execution_time:.2f} seconds")
        logger.info(f"Best model: {automl_results.get('best_model_name', 'Unknown')} with score: {automl_results.get('best_score', 0.0):.4f}")
        logger.info(f"Total models trained: {len(automl_results.get('models', {}))}")
        
    except Exception as e:
        logger.error(f"Error in comprehensive AutoML session {session_id}: {e}")
        
        # Save error status
        try:
            error_folder = Path("data/models/automl") / session_id
            error_folder.mkdir(parents=True, exist_ok=True)
            
            error_status = {
                'session_id': session_id,
                'status': 'error',
                'error_message': str(e),
                'error_timestamp': time.time(),
                'progress': 0
            }
            
            error_file = error_folder / "status.json"
            with open(error_file, 'w') as f:
                json.dump(error_status, f, indent=2)
        except Exception as save_error:
            logger.error(f"Failed to save AutoML error status: {save_error}")
