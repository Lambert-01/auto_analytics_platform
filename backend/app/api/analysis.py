"""Data analysis API endpoints."""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.models.analysis import (
    DataAnalysisRequest,
    DataAnalysisResult,
    AnalysisListResponse,
    AnalysisListItem,
    QuickAnalysisResponse,
    QuickInsight,
    AnalysisType
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/analysis", response_model=dict)
async def start_data_analysis(
    request: DataAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Start data analysis for a dataset.
    
    Args:
        request: Analysis configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Analysis job information
    """
    try:
        logger.info(f"Starting analysis for dataset: {request.dataset_id}")
        
        # Generate analysis ID
        analysis_id = f"analysis_{request.dataset_id}_{int(datetime.now().timestamp())}"
        
        # Start background analysis task
        background_tasks.add_task(
            perform_data_analysis,
            analysis_id,
            request
        )
        
        return {
            "analysis_id": analysis_id,
            "dataset_id": request.dataset_id,
            "status": "started",
            "message": "Data analysis started successfully",
            "analysis_types": request.analysis_types,
            "estimated_completion_time": "5-10 minutes",
            "started_at": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error starting analysis: {e}")
        raise HTTPException(status_code=500, detail="Error starting data analysis")


@router.get("/analysis", response_model=AnalysisListResponse)
async def list_analyses(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    dataset_id: Optional[str] = Query(None, description="Filter by dataset")
):
    """List all data analyses with pagination.
    
    Args:
        page: Page number
        page_size: Items per page
        dataset_id: Filter by dataset ID
        
    Returns:
        Paginated list of analyses
    """
    try:
        # TODO: Implement database queries
        # For now, return sample data
        
        sample_analyses = [
            AnalysisListItem(
                analysis_id="analysis_1",
                dataset_id="dataset_1",
                dataset_filename="sales_data.csv",
                analysis_types=[AnalysisType.DESCRIPTIVE, AnalysisType.CORRELATION],
                data_quality_score=87.5,
                execution_time=125.3,
                generated_at=datetime.now()
            ),
            AnalysisListItem(
                analysis_id="analysis_2",
                dataset_id="dataset_2",
                dataset_filename="customer_data.xlsx",
                analysis_types=[AnalysisType.CLUSTERING, AnalysisType.OUTLIER_DETECTION],
                data_quality_score=92.1,
                execution_time=89.7,
                generated_at=datetime.now()
            )
        ]
        
        # Apply filters
        filtered_analyses = sample_analyses
        if dataset_id:
            filtered_analyses = [a for a in filtered_analyses if a.dataset_id == dataset_id]
        
        return AnalysisListResponse(
            analyses=filtered_analyses,
            total_count=len(filtered_analyses),
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error listing analyses: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving analyses")


@router.get("/analysis/{analysis_id}", response_model=DataAnalysisResult)
async def get_analysis_results(analysis_id: str):
    """Get detailed results of a data analysis.
    
    Args:
        analysis_id: Unique analysis identifier
        
    Returns:
        Complete analysis results
    """
    try:
        # TODO: Implement database query for analysis results
        # For now, return sample data
        
        from app.models.analysis import (
            ColumnAnalysis, StatisticalSummary, CategoricalSummary,
            CorrelationAnalysis, OutlierDetectionResult
        )
        
        sample_column_analyses = [
            ColumnAnalysis(
                column_name="sales_amount",
                data_type="float64",
                is_numerical=True,
                is_categorical=False,
                is_datetime=False,
                numerical_summary=StatisticalSummary(
                    count=10000,
                    mean=125.5,
                    std=45.2,
                    min=10.0,
                    q25=95.0,
                    q50=120.0,
                    q75=155.0,
                    max=500.0,
                    skewness=0.5,
                    kurtosis=2.1,
                    variance=2043.04
                ),
                missing_count=25,
                missing_percentage=0.25,
                unique_count=8500,
                unique_percentage=85.0,
                outlier_count=45,
                outlier_percentage=0.45,
                outlier_method="iqr"
            )
        ]
        
        sample_correlation = CorrelationAnalysis(
            method="pearson",
            correlation_matrix={
                "sales_amount": {"sales_amount": 1.0, "price": 0.85},
                "price": {"sales_amount": 0.85, "price": 1.0}
            },
            strong_correlations=[
                {"column1": "sales_amount", "column2": "price", "correlation": 0.85}
            ],
            correlation_threshold=0.7
        )
        
        sample_outliers = OutlierDetectionResult(
            method="iqr",
            total_outliers=45,
            outlier_percentage=0.45,
            outliers_by_column={"sales_amount": [100, 250, 500, 750, 999]},
            outlier_summary={"sales_amount": 45}
        )
        
        return DataAnalysisResult(
            dataset_id="dataset_1",
            analysis_id=analysis_id,
            dataset_shape=(10000, 5),
            column_analyses=sample_column_analyses,
            correlation_analysis=sample_correlation,
            outlier_detection=sample_outliers,
            key_insights=[
                "Strong positive correlation between sales amount and price",
                "Sales data shows normal distribution with slight right skew",
                "0.45% of sales amounts are potential outliers",
                "Missing values are minimal across all columns"
            ],
            data_quality_score=87.5,
            recommendations=[
                "Investigate high-value sales outliers for accuracy",
                "Consider log transformation for sales amount to reduce skewness",
                "Implement data validation for price entries"
            ],
            analysis_types_performed=[AnalysisType.DESCRIPTIVE, AnalysisType.CORRELATION, AnalysisType.OUTLIER_DETECTION],
            execution_time=125.3
        )
        
    except Exception as e:
        logger.error(f"Error getting analysis results for {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving analysis results")


@router.get("/analysis/{analysis_id}/status")
async def get_analysis_status(analysis_id: str):
    """Get the current status of a data analysis.
    
    Args:
        analysis_id: Unique analysis identifier
        
    Returns:
        Current analysis status and progress
    """
    try:
        # TODO: Implement actual status tracking
        
        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "progress": 100,
            "current_step": "Analysis complete",
            "steps_completed": [
                "Data loading",
                "Descriptive statistics",
                "Correlation analysis",
                "Outlier detection",
                "Report generation"
            ],
            "execution_time": 125.3,
            "estimated_remaining_time": 0,
            "updated_at": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting analysis status for {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving analysis status")


@router.post("/analysis/quick/{dataset_id}", response_model=QuickAnalysisResponse)
async def quick_analysis(dataset_id: str):
    """Perform quick analysis for immediate insights.
    
    Args:
        dataset_id: Unique dataset identifier
        
    Returns:
        Quick analysis results with key insights
    """
    try:
        logger.info(f"Starting quick analysis for dataset: {dataset_id}")
        
        # TODO: Implement actual quick analysis
        
        quick_insights = [
            QuickInsight(
                type="data_quality",
                title="Good Data Quality",
                description="Dataset has minimal missing values (0.25%) and good overall quality",
                importance="medium",
                affected_columns=["sales_amount"],
                suggested_action="Monitor data collection process for consistency"
            ),
            QuickInsight(
                type="correlation",
                title="Strong Price-Sales Relationship",
                description="Sales amount and price show strong positive correlation (0.85)",
                importance="high",
                affected_columns=["sales_amount", "price"],
                suggested_action="Consider price as key feature for sales prediction models"
            ),
            QuickInsight(
                type="distribution",
                title="Normal Distribution with Outliers",
                description="Sales data follows normal distribution but contains potential outliers",
                importance="medium",
                affected_columns=["sales_amount"],
                suggested_action="Review high-value transactions for accuracy"
            )
        ]
        
        summary_stats = {
            "total_rows": 10000,
            "total_columns": 5,
            "numerical_columns": 3,
            "categorical_columns": 2,
            "missing_values_percentage": 0.25,
            "duplicate_rows": 0,
            "data_types": {"float64": 3, "object": 2}
        }
        
        return QuickAnalysisResponse(
            dataset_id=dataset_id,
            quick_insights=quick_insights,
            summary_stats=summary_stats,
            data_quality_score=87.5,
            execution_time=5.2
        )
        
    except Exception as e:
        logger.error(f"Error performing quick analysis for {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail="Error performing quick analysis")


@router.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Delete an analysis and its results.
    
    Args:
        analysis_id: Unique analysis identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        logger.info(f"Deleting analysis: {analysis_id}")
        
        # TODO: Implement actual deletion logic
        
        return {
            "analysis_id": analysis_id,
            "deleted": True,
            "message": "Analysis results deleted successfully",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error deleting analysis {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Error deleting analysis")


async def perform_data_analysis(analysis_id: str, request: DataAnalysisRequest):
    """Background task to perform comprehensive data analysis.
    
    Args:
        analysis_id: Unique analysis identifier
        request: Analysis configuration
    """
    try:
        import time
        import json
        from pathlib import Path
        import pandas as pd
        
        start_time = time.time()
        logger.info(f"Starting comprehensive background analysis: {analysis_id}")
        
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
            logger.error(f"Dataset not found for analysis: {request.dataset_id}")
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
        
        logger.info(f"Dataset loaded: {df.shape}")
        
        # Initialize comprehensive data processor
        from app.core.data_processor import ComprehensiveDataProcessor
        processor = ComprehensiveDataProcessor()
        
        # Perform comprehensive analysis
        processing_results = processor.process_dataset(
            df=df,
            target_column=request.target_column,
            dataset_name=request.dataset_id
        )
        
        # Initialize AI insights engine
        from app.core.ai_insights_engine import AIInsightsEngine
        insights_engine = AIInsightsEngine()
        
        # Generate comprehensive insights
        ai_insights = insights_engine.generate_comprehensive_insights(
            dataset_info=processing_results['dataset_info'],
            profiling_results=processing_results['profiling'],
            analysis_results=processing_results,
            model_results=None
        )
        
        # Perform AutoML analysis if target column is specified
        model_results = None
        if request.target_column and request.target_column in df.columns:
            from app.core.automl_engine import AutoMLEngine
            automl_engine = AutoMLEngine()
            
            try:
                model_results = automl_engine.train_best_model(
                    df=df,
                    target_column=request.target_column,
                    problem_type='auto'  # Auto-detect classification or regression
                )
                logger.info(f"AutoML completed. Best model: {model_results.get('best_model_name', 'Unknown')}")
            except Exception as e:
                logger.warning(f"AutoML failed: {e}")
        
        # Generate comprehensive report
        from app.core.report_generator import ComprehensiveReportGenerator
        report_generator = ComprehensiveReportGenerator()
        
        report_data = report_generator.generate_comprehensive_report(
            dataset_info=processing_results['dataset_info'],
            profiling_results=processing_results['profiling'],
            eda_results=processing_results.get('eda', {}),
            model_results=model_results,
            ai_insights=ai_insights
        )
        
        # Save analysis results
        results_folder = data_folder / "reports" / analysis_id
        results_folder.mkdir(parents=True, exist_ok=True)
        
        # Save detailed analysis results
        analysis_results = {
            'analysis_id': analysis_id,
            'dataset_id': request.dataset_id,
            'dataset_path': str(dataset_path),
            'analysis_types': request.analysis_types,
            'target_column': request.target_column,
            'processing_results': _make_serializable(processing_results),
            'ai_insights': _make_serializable(ai_insights),
            'model_results': _make_serializable(model_results) if model_results else None,
            'report_data': _make_serializable(report_data),
            'execution_time': time.time() - start_time,
            'completed_at': time.time(),
            'status': 'completed'
        }
        
        # Save to JSON file
        results_file = results_folder / "analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Generate and save HTML report if requested
        if report_data.get('html_content'):
            html_file = results_folder / "analysis_report.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(report_data['html_content'])
            logger.info(f"HTML report saved: {html_file}")
        
        # Generate and save PDF report if requested
        if report_data.get('pdf_path'):
            logger.info(f"PDF report saved: {report_data['pdf_path']}")
        
        execution_time = time.time() - start_time
        logger.info(f"Comprehensive analysis completed: {analysis_id} in {execution_time:.2f} seconds")
        
        # Update analysis status (this would normally be in a database)
        status_file = results_folder / "status.json"
        status_data = {
            'analysis_id': analysis_id,
            'status': 'completed',
            'progress': 100,
            'execution_time': execution_time,
            'completed_at': time.time(),
            'results_available': True,
            'insights_count': len(ai_insights.get('key_findings', [])),
            'recommendations_count': len(ai_insights.get('recommendations', [])),
            'quality_score': processing_results.get('profiling', {}).get('overview', {}).get('quality_score', 0)
        }
        
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        
    except Exception as e:
        logger.error(f"Error in comprehensive background analysis {analysis_id}: {e}")
        
        # Save error status
        try:
            error_folder = Path("data/reports") / analysis_id
            error_folder.mkdir(parents=True, exist_ok=True)
            
            error_status = {
                'analysis_id': analysis_id,
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


def _make_serializable(obj):
    """Convert complex objects to JSON-serializable format."""
    import numpy as np
    import pandas as pd
    from datetime import datetime
    
    if isinstance(obj, dict):
        return {key: _make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
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
        return _make_serializable(obj.to_dict())
    elif hasattr(obj, '__dict__'):
        return _make_serializable(obj.__dict__)
    else:
        return obj
