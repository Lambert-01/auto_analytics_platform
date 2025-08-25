"""Visualization generation API endpoints."""

from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/visualizations/generate")
async def generate_visualization(
    dataset_id: str,
    chart_type: str,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    color_column: Optional[str] = None,
    title: Optional[str] = None,
    width: int = 800,
    height: int = 600,
    interactive: bool = True
):
    """Generate a visualization for dataset.
    
    Args:
        dataset_id: Unique dataset identifier
        chart_type: Type of chart (bar, line, scatter, histogram, etc.)
        x_column: Column for x-axis
        y_column: Column for y-axis
        color_column: Column for color encoding
        title: Chart title
        width: Chart width in pixels
        height: Chart height in pixels
        interactive: Whether to generate interactive chart
        
    Returns:
        Visualization information and file path
    """
    try:
        from pathlib import Path
        import pandas as pd
        
        logger.info(f"Generating {chart_type} visualization for dataset: {dataset_id}")
        
        # Find and load the dataset
        dataset_path = None
        data_folder = Path("data")
        
        for folder in ["uploads", "processed"]:
            folder_path = data_folder / folder
            if folder_path.exists():
                for file_path in folder_path.glob(f"*{dataset_id}*"):
                    if file_path.suffix in ['.csv', '.xlsx', '.parquet']:
                        dataset_path = file_path
                        break
        
        if not dataset_path:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Load dataset
        if dataset_path.suffix == '.csv':
            df = pd.read_csv(dataset_path)
        elif dataset_path.suffix == '.xlsx':
            df = pd.read_excel(dataset_path)
        elif dataset_path.suffix == '.parquet':
            df = pd.read_parquet(dataset_path)
        
        # Validate columns exist
        if x_column and x_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{x_column}' not found in dataset")
        if y_column and y_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{y_column}' not found in dataset")
        if color_column and color_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{color_column}' not found in dataset")
        
        # Use comprehensive data processor for visualization
        from app.core.data_processor import ComprehensiveDataProcessor
        processor = ComprehensiveDataProcessor()
        
        # Generate visualization using the processor
        chart_id = f"chart_{dataset_id}_{int(datetime.now().timestamp())}"
        
        visualization_result = processor.generate_custom_visualization(
            df=df,
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column,
            color_column=color_column,
            title=title,
            width=width,
            height=height,
            interactive=interactive,
            save_path=str(data_folder / "charts" / f"{chart_id}")
        )
        
        if not visualization_result.get('success'):
            raise HTTPException(status_code=500, detail=visualization_result.get('error', 'Visualization generation failed'))
        
        file_path = visualization_result['file_path']
        actual_file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
        
        return {
            "chart_id": chart_id,
            "dataset_id": dataset_id,
            "chart_type": chart_type,
            "file_path": file_path,
            "interactive": interactive,
            "dimensions": {"width": width, "height": height},
            "columns_used": {
                "x_column": x_column,
                "y_column": y_column,
                "color_column": color_column
            },
            "title": visualization_result.get('title', title or f"{chart_type.title()} Chart"),
            "generated_at": datetime.now(),
            "file_size": actual_file_size,
            "status": "completed",
            "chart_data": visualization_result.get('chart_data', {}),
            "chart_config": visualization_result.get('chart_config', {})
        }
        
    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        raise HTTPException(status_code=500, detail="Error generating visualization")


@router.post("/visualizations/auto-generate/{dataset_id}")
async def auto_generate_visualizations(dataset_id: str):
    """Automatically generate appropriate visualizations for a dataset.
    
    Args:
        dataset_id: Unique dataset identifier
        
    Returns:
        List of generated visualizations
    """
    try:
        from pathlib import Path
        import pandas as pd
        
        logger.info(f"Auto-generating visualizations for dataset: {dataset_id}")
        
        # Find and load the dataset
        dataset_path = None
        data_folder = Path("data")
        
        for folder in ["uploads", "processed"]:
            folder_path = data_folder / folder
            if folder_path.exists():
                for file_path in folder_path.glob(f"*{dataset_id}*"):
                    if file_path.suffix in ['.csv', '.xlsx', '.parquet']:
                        dataset_path = file_path
                        break
        
        if not dataset_path:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Load dataset
        if dataset_path.suffix == '.csv':
            df = pd.read_csv(dataset_path)
        elif dataset_path.suffix == '.xlsx':
            df = pd.read_excel(dataset_path)
        elif dataset_path.suffix == '.parquet':
            df = pd.read_parquet(dataset_path)
        
        # Use comprehensive data processor for auto-visualization
        from app.core.data_processor import ComprehensiveDataProcessor
        processor = ComprehensiveDataProcessor()
        
        # Generate automatic visualizations based on data analysis
        auto_viz_results = processor.generate_automatic_visualizations(
            df=df,
            dataset_name=dataset_id,
            save_folder=str(data_folder / "charts" / f"auto_{dataset_id}")
        )
        
        if not auto_viz_results.get('success'):
            raise HTTPException(status_code=500, detail=auto_viz_results.get('error', 'Auto-visualization generation failed'))
        
        visualizations = []
        for viz in auto_viz_results.get('visualizations', []):
            chart_info = {
                "chart_id": viz.get('chart_id'),
                "chart_type": viz.get('chart_type'),
                "title": viz.get('title'),
                "description": viz.get('description'),
                "columns": viz.get('columns', []),
                "file_path": viz.get('file_path'),
                "importance": viz.get('importance', 'medium'),
                "insights": viz.get('insights', []),
                "relevance_score": viz.get('relevance_score', 0.5),
                "file_size": Path(viz['file_path']).stat().st_size if Path(viz['file_path']).exists() else 0
            }
            visualizations.append(chart_info)
        
        return {
            "dataset_id": dataset_id,
            "total_visualizations": len(visualizations),
            "visualizations": visualizations,
            "generated_at": datetime.now(),
            "status": "completed",
            "generation_time": auto_viz_results.get('generation_time', 0),
            "data_analysis": auto_viz_results.get('data_analysis', {}),
            "recommendations": auto_viz_results.get('recommendations', [])
        }
        
    except Exception as e:
        logger.error(f"Error auto-generating visualizations for {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail="Error auto-generating visualizations")


@router.get("/visualizations")
async def list_visualizations(
    dataset_id: Optional[str] = Query(None, description="Filter by dataset"),
    chart_type: Optional[str] = Query(None, description="Filter by chart type"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page")
):
    """List all generated visualizations.
    
    Args:
        dataset_id: Filter by dataset ID
        chart_type: Filter by chart type
        page: Page number
        page_size: Items per page
        
    Returns:
        Paginated list of visualizations
    """
    try:
        # TODO: Implement database queries
        # For now, return sample data
        
        sample_visualizations = [
            {
                "chart_id": "chart_1",
                "dataset_id": "dataset_1",
                "chart_type": "bar",
                "title": "Sales by Category",
                "file_path": "data/charts/chart_1.html",
                "interactive": True,
                "generated_at": datetime.now(),
                "file_size": 256000
            },
            {
                "chart_id": "chart_2", 
                "dataset_id": "dataset_1",
                "chart_type": "line",
                "title": "Sales Trend Over Time",
                "file_path": "data/charts/chart_2.html",
                "interactive": True,
                "generated_at": datetime.now(),
                "file_size": 312000
            }
        ]
        
        # Apply filters
        filtered_visualizations = sample_visualizations
        if dataset_id:
            filtered_visualizations = [v for v in filtered_visualizations if v["dataset_id"] == dataset_id]
        if chart_type:
            filtered_visualizations = [v for v in filtered_visualizations if v["chart_type"] == chart_type]
        
        return {
            "visualizations": filtered_visualizations,
            "total_count": len(filtered_visualizations),
            "page": page,
            "page_size": page_size
        }
        
    except Exception as e:
        logger.error(f"Error listing visualizations: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving visualizations")


@router.get("/visualizations/{chart_id}")
async def get_visualization_details(chart_id: str):
    """Get detailed information about a visualization.
    
    Args:
        chart_id: Unique chart identifier
        
    Returns:
        Detailed visualization information
    """
    try:
        # TODO: Implement database query
        # For now, return sample data
        
        return {
            "chart_id": chart_id,
            "dataset_id": "dataset_1",
            "chart_type": "bar",
            "title": "Sales by Category",
            "description": "Bar chart showing total sales amount by product category",
            "file_path": "data/charts/chart_1.html",
            "interactive": True,
            "dimensions": {"width": 800, "height": 600},
            "columns_used": {
                "x_column": "category",
                "y_column": "sales_amount",
                "color_column": None
            },
            "styling": {
                "color_scheme": "viridis",
                "theme": "modern"
            },
            "generated_at": datetime.now(),
            "file_size": 256000,
            "download_count": 5,
            "insights": [
                "Electronics category has the highest total sales",
                "Clothing and Books categories have similar sales volumes"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting visualization details for {chart_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving visualization details")


@router.get("/visualizations/{chart_id}/download")
async def download_visualization(chart_id: str, format: str = Query("html", pattern="^(html|png|svg|pdf)$")):
    """Download a visualization file.
    
    Args:
        chart_id: Unique chart identifier
        format: Download format (html, png, svg, pdf)
        
    Returns:
        File download response
    """
    try:
        # TODO: Implement actual file serving
        # For now, return placeholder response
        
        filename = f"{chart_id}.{format}"
        file_path = f"data/charts/{filename}"
        
        # In a real implementation, this would serve the actual file
        return {
            "chart_id": chart_id,
            "format": format,
            "filename": filename,
            "download_url": f"/static/charts/{filename}",
            "file_size": 256000,
            "generated_at": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error downloading visualization {chart_id}: {e}")
        raise HTTPException(status_code=500, detail="Error downloading visualization")


@router.delete("/visualizations/{chart_id}")
async def delete_visualization(chart_id: str):
    """Delete a visualization.
    
    Args:
        chart_id: Unique chart identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        logger.info(f"Deleting visualization: {chart_id}")
        
        # TODO: Implement actual deletion logic
        
        return {
            "chart_id": chart_id,
            "deleted": True,
            "message": "Visualization deleted successfully",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error deleting visualization {chart_id}: {e}")
        raise HTTPException(status_code=500, detail="Error deleting visualization")


@router.get("/visualizations/types/available")
async def get_available_chart_types():
    """Get list of available chart types and their descriptions.
    
    Returns:
        Available chart types with descriptions and use cases
    """
    try:
        chart_types = {
            "bar": {
                "name": "Bar Chart",
                "description": "Compare values across categories",
                "use_cases": ["Categorical data comparison", "Ranking data"],
                "required_columns": ["categorical"],
                "optional_columns": ["numerical", "color"]
            },
            "line": {
                "name": "Line Chart",
                "description": "Show trends over time or continuous variables",
                "use_cases": ["Time series data", "Trend analysis"],
                "required_columns": ["continuous"],
                "optional_columns": ["numerical", "color"]
            },
            "scatter": {
                "name": "Scatter Plot",
                "description": "Show relationship between two numerical variables",
                "use_cases": ["Correlation analysis", "Pattern detection"],
                "required_columns": ["numerical", "numerical"],
                "optional_columns": ["color", "size"]
            },
            "histogram": {
                "name": "Histogram",
                "description": "Show distribution of numerical data",
                "use_cases": ["Data distribution", "Outlier detection"],
                "required_columns": ["numerical"],
                "optional_columns": ["color"]
            },
            "box_plot": {
                "name": "Box Plot",
                "description": "Show statistical summary and outliers",
                "use_cases": ["Statistical analysis", "Outlier detection", "Group comparison"],
                "required_columns": ["numerical"],
                "optional_columns": ["categorical"]
            },
            "heatmap": {
                "name": "Heatmap",
                "description": "Show correlation matrix or 2D data patterns",
                "use_cases": ["Correlation analysis", "Pattern visualization"],
                "required_columns": ["numerical_matrix"],
                "optional_columns": []
            },
            "pie": {
                "name": "Pie Chart",
                "description": "Show proportions of categorical data",
                "use_cases": ["Proportion analysis", "Composition breakdown"],
                "required_columns": ["categorical"],
                "optional_columns": ["numerical"]
            }
        }
        
        return {
            "chart_types": chart_types,
            "total_types": len(chart_types),
            "categories": ["Statistical", "Comparison", "Distribution", "Relationship"]
        }
        
    except Exception as e:
        logger.error(f"Error getting chart types: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving chart types")
