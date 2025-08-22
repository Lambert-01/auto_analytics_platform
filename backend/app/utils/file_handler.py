"""File handling utilities for data uploads and processing."""

import hashlib
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import aiofiles
import pandas as pd
from fastapi import HTTPException, UploadFile

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class FileHandler:
    """Handle file operations for the analytics platform."""
    
    def __init__(self):
        """Initialize file handler with settings."""
        self.upload_dir = settings.upload_dir
        self.processed_dir = settings.processed_dir
        self.cache_dir = settings.cache_dir
        self.max_file_size = settings.max_file_size
        self.allowed_extensions = settings.allowed_extensions
    
    async def save_uploaded_file(self, file: UploadFile) -> Dict[str, str]:
        """Save uploaded file and return file information.
        
        Args:
            file: Uploaded file object
            
        Returns:
            Dictionary containing file information
            
        Raises:
            HTTPException: If file is invalid or save fails
        """
        try:
            # Validate file
            self._validate_file(file)
            
            # Generate unique filename
            file_id = str(uuid4())
            file_extension = Path(file.filename).suffix.lower()
            filename = f"{file_id}{file_extension}"
            file_path = self.upload_dir / filename
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)
            
            # Get file info
            file_info = {
                "file_id": file_id,
                "filename": filename,
                "original_filename": file.filename,
                "file_path": str(file_path),
                "file_size": len(content),
                "file_hash": file_hash,
                "content_type": file.content_type,
                "extension": file_extension
            }
            
            logger.info(f"File saved successfully: {filename}")
            return file_info
            
        except Exception as e:
            logger.error(f"Failed to save file {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    def _validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file.
        
        Args:
            file: Uploaded file object
            
        Raises:
            HTTPException: If file is invalid
        """
        # Check if file exists
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in self.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not allowed. Allowed types: {self.allowed_extensions}"
            )
        
        # Check file size (if available)
        if hasattr(file, 'size') and file.size and file.size > self.max_file_size:
            max_size_mb = self.max_file_size / (1024 * 1024)
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {max_size_mb:.1f}MB"
            )
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal hash string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load dataset from file path.
        
        Args:
            file_path: Path to dataset file
            
        Returns:
            Pandas DataFrame
            
        Raises:
            HTTPException: If file cannot be loaded
        """
        try:
            file_path = Path(file_path)
            extension = file_path.suffix.lower()
            
            if extension == '.csv':
                df = pd.read_csv(file_path)
            elif extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif extension == '.json':
                df = pd.read_json(file_path)
            elif extension == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
            
            logger.info(f"Dataset loaded successfully: {file_path.name}, shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load dataset {file_path}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to load dataset: {str(e)}")
    
    def save_processed_dataset(self, df: pd.DataFrame, dataset_id: str, format: str = 'parquet') -> str:
        """Save processed dataset.
        
        Args:
            df: Processed DataFrame
            dataset_id: Unique dataset identifier
            format: Output format ('parquet', 'csv')
            
        Returns:
            Path to saved file
        """
        try:
            if format == 'parquet':
                filename = f"{dataset_id}_processed.parquet"
                file_path = self.processed_dir / filename
                df.to_parquet(file_path)
            elif format == 'csv':
                filename = f"{dataset_id}_processed.csv"
                file_path = self.processed_dir / filename
                df.to_csv(file_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Processed dataset saved: {filename}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save processed dataset: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save processed dataset: {str(e)}")
    
    def delete_file(self, file_path: str) -> bool:
        """Delete file from filesystem.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"File deleted: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False
    
    def get_file_info(self, file_path: str) -> Dict[str, any]:
        """Get information about a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            stat = file_path.stat()
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            return {
                "name": file_path.name,
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "extension": file_path.suffix.lower(),
                "mime_type": mime_type,
                "exists": True
            }
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            return {"exists": False, "error": str(e)}
    
    def list_uploaded_files(self) -> List[Dict[str, any]]:
        """List all uploaded files.
        
        Returns:
            List of file information dictionaries
        """
        try:
            files = []
            for file_path in self.upload_dir.glob("*"):
                if file_path.is_file():
                    file_info = self.get_file_info(str(file_path))
                    files.append(file_info)
            
            logger.info(f"Found {len(files)} uploaded files")
            return files
        except Exception as e:
            logger.error(f"Failed to list uploaded files: {e}")
            return []


# Global file handler instance
file_handler = FileHandler()
