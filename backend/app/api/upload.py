"""File upload API endpoints."""

import asyncio
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse

from app.models.dataset import (
    DatasetUploadResponse, 
    DatasetStatus, 
    DatasetListResponse,
    DatasetListItem
)
from app.utils.file_handler import file_handler
from app.utils.logger import get_logger
from app.config import settings

logger = get_logger(__name__)
router = APIRouter()


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload a dataset file.
    
    Args:
        file: Uploaded file
        background_tasks: FastAPI background tasks
        
    Returns:
        Upload response with file information
    """
    try:
        logger.info(f"Received file upload request: {file.filename}")
        
        # Save uploaded file
        file_info = await file_handler.save_uploaded_file(file)
        
        # Start background processing
        background_tasks.add_task(
            process_uploaded_dataset,
            file_info["file_id"],
            file_info["file_path"]
        )
        
        response = DatasetUploadResponse(
            file_id=file_info["file_id"],
            filename=file_info["filename"],
            file_size=file_info["file_size"],
            status=DatasetStatus.UPLOADED,
            message="File uploaded successfully. Processing started.",
            upload_timestamp=datetime.now()
        )
        
        logger.info(f"File upload completed: {file_info['filename']}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during file upload: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during upload")


@router.post("/upload/multiple", response_model=List[DatasetUploadResponse])
async def upload_multiple_datasets(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Upload multiple dataset files.
    
    Args:
        files: List of uploaded files
        background_tasks: FastAPI background tasks
        
    Returns:
        List of upload responses
    """
    try:
        if len(files) > 10:  # Limit to 10 files at once
            raise HTTPException(
                status_code=400, 
                detail="Maximum 10 files can be uploaded at once"
            )
        
        responses = []
        
        for file in files:
            try:
                logger.info(f"Processing file: {file.filename}")
                
                # Save uploaded file
                file_info = await file_handler.save_uploaded_file(file)
                
                # Start background processing
                background_tasks.add_task(
                    process_uploaded_dataset,
                    file_info["file_id"],
                    file_info["file_path"]
                )
                
                response = DatasetUploadResponse(
                    file_id=file_info["file_id"],
                    filename=file_info["filename"],
                    file_size=file_info["file_size"],
                    status=DatasetStatus.UPLOADED,
                    message="File uploaded successfully. Processing started.",
                    upload_timestamp=datetime.now()
                )
                responses.append(response)
                
            except Exception as e:
                logger.error(f"Failed to upload file {file.filename}: {e}")
                error_response = DatasetUploadResponse(
                    file_id="",
                    filename=file.filename or "unknown",
                    file_size=0,
                    status=DatasetStatus.ERROR,
                    message=f"Upload failed: {str(e)}",
                    upload_timestamp=datetime.now()
                )
                responses.append(error_response)
        
        logger.info(f"Multiple file upload completed: {len(responses)} files processed")
        return responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during multiple file upload: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during upload")


@router.get("/upload/status/{file_id}")
async def get_upload_status(file_id: str):
    """Get upload and processing status for a file.
    
    Args:
        file_id: Unique file identifier
        
    Returns:
        Current status information
    """
    try:
        # This would normally query a database for the current status
        # For now, we'll return a basic response
        # TODO: Implement database queries
        
        return {
            "file_id": file_id,
            "status": "processing",
            "message": "File is being processed",
            "progress": 75,
            "estimated_completion_time": "2 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error getting upload status for {file_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving upload status")


@router.delete("/upload/{file_id}")
async def delete_uploaded_file(file_id: str):
    """Delete an uploaded file.
    
    Args:
        file_id: Unique file identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        # TODO: Implement file deletion logic with database cleanup
        logger.info(f"Delete request for file: {file_id}")
        
        return {
            "file_id": file_id,
            "deleted": True,
            "message": "File deleted successfully",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error deleting file {file_id}: {e}")
        raise HTTPException(status_code=500, detail="Error deleting file")


@router.get("/upload/list", response_model=DatasetListResponse)
async def list_uploaded_files(
    page: int = 1,
    page_size: int = 50,
    status: str = None
):
    """List uploaded files with pagination.
    
    Args:
        page: Page number (1-based)
        page_size: Number of items per page
        status: Filter by status (optional)
        
    Returns:
        Paginated list of uploaded files
    """
    try:
        # TODO: Implement database queries for listing files
        # For now, return sample data
        
        sample_files = [
            DatasetListItem(
                file_id="sample-1",
                filename="sample_data.csv",
                original_filename="sales_data.csv",
                file_size=1024000,
                rows=5000,
                columns=12,
                dataset_type=None,
                status=DatasetStatus.PROCESSED,
                upload_timestamp=datetime.now(),
                processing_duration=45.2
            )
        ]
        
        return DatasetListResponse(
            datasets=sample_files,
            total_count=len(sample_files),
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error listing uploaded files: {e}")
        raise HTTPException(status_code=500, detail="Error listing files")


async def process_uploaded_dataset(file_id: str, file_path: str):
    """Background task to process uploaded dataset.
    
    Args:
        file_id: Unique file identifier
        file_path: Path to uploaded file
    """
    try:
        logger.info(f"Starting comprehensive processing for file: {file_id}")
        
        # Load dataset
        df = file_handler.load_dataset(file_path)
        logger.info(f"Dataset loaded: {df.shape}")
        
        # Initialize comprehensive data processor
        from app.core.data_processor import ComprehensiveDataProcessor
        processor = ComprehensiveDataProcessor()
        
        # Perform comprehensive analysis
        processing_results = processor.process_dataset(
            df=df,
            target_column=None,  # Auto-detect or user can specify later
            dataset_name=file_id
        )
        
        # Save processed data
        processed_path = file_handler.save_processed_dataset(df, file_id)
        logger.info(f"Processed dataset saved: {processed_path}")
        
        # Save analysis results
        results_path = processed_path.replace('.parquet', '_analysis.json')
        with open(results_path, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = _make_json_serializable(processing_results)
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Analysis results saved: {results_path}")
        
        # Generate insights
        from app.core.ai_insights_engine import AIInsightsEngine
        insights_engine = AIInsightsEngine()
        
        ai_insights = insights_engine.generate_comprehensive_insights(
            dataset_info=processing_results['dataset_info'],
            profiling_results=processing_results['profiling'],
            analysis_results=processing_results,
            model_results=None
        )
        
        # Save insights
        insights_path = processed_path.replace('.parquet', '_insights.json')
        with open(insights_path, 'w') as f:
            json.dump(_make_json_serializable(ai_insights), f, indent=2)
        
        logger.info(f"AI insights generated and saved: {insights_path}")
        
        # Store metadata for future retrieval
        metadata = {
            'file_id': file_id,
            'original_path': file_path,
            'processed_path': processed_path,
            'analysis_path': results_path,
            'insights_path': insights_path,
            'shape': df.shape,
            'processing_timestamp': datetime.now().isoformat(),
            'status': 'completed',
            'quality_score': processing_results.get('profiling', {}).get('overview', {}).get('quality_score', 0),
            'summary': {
                'total_insights': len(ai_insights.get('key_findings', [])),
                'data_quality': 'excellent' if processing_results.get('profiling', {}).get('overview', {}).get('missing_data', {}).get('missing_percentage', 0) < 5 else 'good',
                'recommendation_count': len(ai_insights.get('recommendations', []))
            }
        }
        
        metadata_path = processed_path.replace('.parquet', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Comprehensive processing completed for file: {file_id}")
        
    except Exception as e:
        logger.error(f"Error processing dataset {file_id}: {e}")
        # Save error information
        error_metadata = {
            'file_id': file_id,
            'status': 'error',
            'error_message': str(e),
            'error_timestamp': datetime.now().isoformat()
        }
        
        error_path = file_path.replace('.csv', '_error.json').replace('.xlsx', '_error.json')
        with open(error_path, 'w') as f:
            json.dump(error_metadata, f, indent=2)


def _make_json_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON-serializable format."""
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
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
    else:
        return obj
