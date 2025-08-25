"""Report generation API endpoints."""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse

from app.models.report import (
    ReportGenerationRequest,
    ReportGenerationResponse,
    ReportListResponse,
    ReportListItem,
    ReportContent,
    ReportMetadata,
    ReportDownloadResponse,
    CustomReportRequest,
    ReportTemplateRequest,
    ReportTemplateResponse,
    ReportAnalytics,
    ReportType,
    ReportFormat,
    ReportStatus
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/reports/generate", response_model=ReportGenerationResponse)
async def generate_report(
    request: ReportGenerationRequest,
    background_tasks: BackgroundTasks
):
    """Generate a comprehensive report.
    
    Args:
        request: Report generation configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Report generation job information
    """
    try:
        logger.info(f"Starting report generation: {request.title}")
        
        # Generate report ID
        report_id = f"report_{int(datetime.now().timestamp())}"
        
        # Start background report generation task
        background_tasks.add_task(
            generate_comprehensive_report,
            report_id,
            request
        )
        
        return ReportGenerationResponse(
            report_id=report_id,
            title=request.title,
            status=ReportStatus.PENDING,
            message="Report generation started successfully",
            estimated_generation_time=180,  # 3 minutes
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error starting report generation: {e}")
        raise HTTPException(status_code=500, detail="Error starting report generation")


@router.get("/reports", response_model=ReportListResponse)
async def list_reports(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    report_type: Optional[ReportType] = Query(None, description="Filter by report type"),
    format: Optional[ReportFormat] = Query(None, description="Filter by format"),
    status: Optional[ReportStatus] = Query(None, description="Filter by status")
):
    """List all generated reports with pagination.
    
    Args:
        page: Page number
        page_size: Items per page
        report_type: Filter by report type
        format: Filter by format
        status: Filter by status
        
    Returns:
        Paginated list of reports
    """
    try:
        # TODO: Implement database queries
        # For now, return sample data
        
        sample_reports = [
            ReportListItem(
                report_id="report_1",
                title="Sales Data Analysis Report",
                report_type=ReportType.COMPLETE_ANALYTICS,
                format=ReportFormat.HTML,
                status=ReportStatus.COMPLETED,
                file_size=2048000,
                generation_duration=156.7,
                created_at=datetime.now()
            ),
            ReportListItem(
                report_id="report_2",
                title="Customer Segmentation Report",
                report_type=ReportType.ML_MODEL_REPORT,
                format=ReportFormat.PDF,
                status=ReportStatus.COMPLETED,
                file_size=1536000,
                generation_duration=89.3,
                created_at=datetime.now()
            )
        ]
        
        # Apply filters
        filtered_reports = sample_reports
        if report_type:
            filtered_reports = [r for r in filtered_reports if r.report_type == report_type]
        if format:
            filtered_reports = [r for r in filtered_reports if r.format == format]
        if status:
            filtered_reports = [r for r in filtered_reports if r.status == status]
        
        return ReportListResponse(
            reports=filtered_reports,
            total_count=len(filtered_reports),
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error listing reports: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving reports")


@router.get("/reports/{report_id}", response_model=ReportContent)
async def get_report_details(report_id: str):
    """Get detailed information about a specific report.
    
    Args:
        report_id: Unique report identifier
        
    Returns:
        Detailed report content and metadata
    """
    try:
        # TODO: Implement database query for report content
        # For now, return sample data
        
        from app.models.report import ExecutiveSummary, DataProfileSummary, ReportSection
        
        sample_metadata = ReportMetadata(
            report_id=report_id,
            title="Sales Data Analysis Report",
            description="Comprehensive analysis of sales data with insights and recommendations",
            report_type=ReportType.COMPLETE_ANALYTICS,
            format=ReportFormat.HTML,
            dataset_id="dataset_1",
            status=ReportStatus.COMPLETED,
            generation_start_time=datetime.now(),
            generation_end_time=datetime.now(),
            generation_duration=156.7,
            file_path=f"reports/html/{report_id}.html",
            file_size=2048000
        )
        
        sample_executive_summary = ExecutiveSummary(
            dataset_overview="Sales dataset contains 10,000 records with 5 key attributes including sales amount, product category, and customer information.",
            key_findings=[
                "Strong positive correlation (0.85) between price and sales amount",
                "Electronics category drives 40% of total sales revenue",
                "Seasonal patterns detected with Q4 showing highest performance"
            ],
            data_quality_assessment="Dataset quality is excellent with minimal missing values (0.25%) and no significant data quality issues.",
            recommendations=[
                "Focus marketing efforts on electronics category",
                "Implement dynamic pricing strategy based on demand patterns",
                "Investigate outlier transactions for potential process improvements"
            ],
            risk_factors=[
                "Heavy dependency on electronics category creates revenue concentration risk",
                "Seasonal fluctuations may impact cash flow stability"
            ]
        )
        
        sample_data_profile = DataProfileSummary(
            total_rows=10000,
            total_columns=5,
            data_types_summary={"numerical": 3, "categorical": 2},
            missing_data_summary={"sales_amount": 0.25, "category": 0.0},
            data_quality_score=87.5,
            top_correlations=[
                {"column1": "price", "column2": "sales_amount", "correlation": 0.85}
            ],
            outlier_summary={"sales_amount": 45},
            data_distribution_insights=[
                "Sales amount follows normal distribution with slight right skew",
                "Product categories are well balanced across the dataset"
            ]
        )
        
        sample_sections = [
            ReportSection(
                section_id="data_overview",
                title="Data Overview",
                content_type="statistics",
                content={
                    "rows": 10000,
                    "columns": 5,
                    "data_quality_score": 87.5
                },
                order=1
            ),
            ReportSection(
                section_id="correlation_analysis",
                title="Correlation Analysis",
                content_type="chart",
                content={
                    "chart_type": "heatmap",
                    "title": "Feature Correlation Matrix"
                },
                order=2
            )
        ]
        
        return ReportContent(
            metadata=sample_metadata,
            executive_summary=sample_executive_summary,
            data_profile=sample_data_profile,
            sections=sample_sections,
            insights=[
                "Electronics category shows strongest sales performance",
                "Price is the primary driver of sales amount",
                "Data quality is excellent for reliable analysis"
            ],
            recommendations=[
                "Expand electronics product line",
                "Implement targeted pricing strategies",
                "Maintain current data collection standards"
            ]
        )
        
    except Exception as e:
        logger.error(f"Error getting report details for {report_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving report details")


@router.get("/reports/{report_id}/download", response_model=ReportDownloadResponse)
async def download_report(report_id: str):
    """Download a generated report.
    
    Args:
        report_id: Unique report identifier
        
    Returns:
        Report download information
    """
    try:
        # TODO: Implement actual file serving
        # For now, return download information
        
        return ReportDownloadResponse(
            report_id=report_id,
            filename=f"report_{report_id}.html",
            file_path=f"reports/html/{report_id}.html",
            file_size=2048000,
            content_type="text/html",
            generated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error downloading report {report_id}: {e}")
        raise HTTPException(status_code=500, detail="Error downloading report")


@router.get("/reports/{report_id}/status")
async def get_report_status(report_id: str):
    """Get the current status of report generation.
    
    Args:
        report_id: Unique report identifier
        
    Returns:
        Current generation status and progress
    """
    try:
        # TODO: Implement actual status tracking
        
        return {
            "report_id": report_id,
            "status": "completed",
            "progress": 100,
            "current_step": "Report generation complete",
            "steps_completed": [
                "Data analysis",
                "Chart generation",
                "Insight extraction",
                "Report compilation",
                "File generation"
            ],
            "generation_time": 156.7,
            "estimated_remaining_time": 0,
            "file_size": 2048000,
            "updated_at": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting report status for {report_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving report status")


@router.post("/reports/custom", response_model=ReportGenerationResponse)
async def generate_custom_report(
    request: CustomReportRequest,
    background_tasks: BackgroundTasks
):
    """Generate a custom report with user-defined sections.
    
    Args:
        request: Custom report configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Report generation job information
    """
    try:
        logger.info(f"Starting custom report generation: {request.title}")
        
        # Generate report ID
        report_id = f"custom_report_{int(datetime.now().timestamp())}"
        
        # Start background custom report generation task
        background_tasks.add_task(
            generate_custom_report_content,
            report_id,
            request
        )
        
        return ReportGenerationResponse(
            report_id=report_id,
            title=request.title,
            status=ReportStatus.PENDING,
            message="Custom report generation started successfully",
            estimated_generation_time=240,  # 4 minutes for custom reports
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error starting custom report generation: {e}")
        raise HTTPException(status_code=500, detail="Error starting custom report generation")


@router.post("/reports/templates", response_model=ReportTemplateResponse)
async def create_report_template(request: ReportTemplateRequest):
    """Create a new report template.
    
    Args:
        request: Report template configuration
        
    Returns:
        Created template information
    """
    try:
        logger.info(f"Creating report template: {request.name}")
        
        # Generate template ID
        template_id = f"template_{int(datetime.now().timestamp())}"
        
        # TODO: Implement template creation logic
        
        return ReportTemplateResponse(
            template_id=template_id,
            name=request.name,
            description=request.description,
            report_type=request.report_type,
            sections_count=len(request.sections),
            is_public=request.is_public,
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error creating report template: {e}")
        raise HTTPException(status_code=500, detail="Error creating report template")


@router.get("/reports/templates")
async def list_report_templates(
    report_type: Optional[ReportType] = Query(None, description="Filter by report type"),
    is_public: Optional[bool] = Query(None, description="Filter by public/private")
):
    """List available report templates.
    
    Args:
        report_type: Filter by report type
        is_public: Filter by public/private status
        
    Returns:
        List of available templates
    """
    try:
        # TODO: Implement database queries for templates
        
        sample_templates = [
            {
                "template_id": "template_1",
                "name": "Standard Data Analysis",
                "description": "Comprehensive data analysis with standard sections",
                "report_type": ReportType.DATA_PROFILE,
                "sections_count": 8,
                "is_public": True,
                "created_at": datetime.now()
            },
            {
                "template_id": "template_2",
                "name": "Executive Summary",
                "description": "High-level overview for business stakeholders",
                "report_type": ReportType.EXECUTIVE_SUMMARY,
                "sections_count": 4,
                "is_public": True,
                "created_at": datetime.now()
            }
        ]
        
        # Apply filters
        filtered_templates = sample_templates
        if report_type:
            filtered_templates = [t for t in filtered_templates if t["report_type"] == report_type]
        if is_public is not None:
            filtered_templates = [t for t in filtered_templates if t["is_public"] == is_public]
        
        return {
            "templates": filtered_templates,
            "total_count": len(filtered_templates)
        }
        
    except Exception as e:
        logger.error(f"Error listing report templates: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving report templates")


@router.get("/reports/analytics", response_model=ReportAnalytics)
async def get_report_analytics():
    """Get analytics about report generation and usage.
    
    Returns:
        Report analytics and statistics
    """
    try:
        # TODO: Implement actual analytics queries
        
        return ReportAnalytics(
            total_reports_generated=150,
            reports_by_type={
                ReportType.DATA_PROFILE: 45,
                ReportType.COMPLETE_ANALYTICS: 38,
                ReportType.ML_MODEL_REPORT: 32,
                ReportType.EXECUTIVE_SUMMARY: 25,
                ReportType.CUSTOM: 10
            },
            reports_by_format={
                ReportFormat.HTML: 85,
                ReportFormat.PDF: 65
            },
            average_generation_time=145.6,
            most_popular_templates=[
                {"template_id": "template_1", "name": "Standard Data Analysis", "usage_count": 45},
                {"template_id": "template_2", "name": "Executive Summary", "usage_count": 25}
            ],
            recent_activity=[
                {"action": "report_generated", "report_id": "report_123", "timestamp": datetime.now()},
                {"action": "template_created", "template_id": "template_5", "timestamp": datetime.now()}
            ],
            generation_date_range=(datetime.now(), datetime.now())
        )
        
    except Exception as e:
        logger.error(f"Error getting report analytics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving report analytics")


@router.delete("/reports/{report_id}")
async def delete_report(report_id: str):
    """Delete a report and its files.
    
    Args:
        report_id: Unique report identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        logger.info(f"Deleting report: {report_id}")
        
        # TODO: Implement actual deletion logic
        
        return {
            "report_id": report_id,
            "deleted": True,
            "message": "Report deleted successfully",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error deleting report {report_id}: {e}")
        raise HTTPException(status_code=500, detail="Error deleting report")


async def generate_comprehensive_report(report_id: str, request: ReportGenerationRequest):
    """Background task to generate comprehensive report.
    
    Args:
        report_id: Unique report identifier
        request: Report generation configuration
    """
    try:
        import time
        import json
        from pathlib import Path
        import pandas as pd
        
        start_time = time.time()
        logger.info(f"Starting comprehensive background report generation: {report_id}")
        
        # Find and load datasets
        data_folder = Path("data")
        datasets = {}
        
        for dataset_id in request.datasets:
            dataset_path = None
            
            # Search for dataset files
            for folder in ["uploads", "processed"]:
                folder_path = data_folder / folder
                if folder_path.exists():
                    for file_path in folder_path.glob(f"*{dataset_id}*"):
                        if file_path.suffix in ['.csv', '.xlsx', '.parquet']:
                            dataset_path = file_path
                            break
            
            if dataset_path:
                # Load dataset
                if dataset_path.suffix == '.csv':
                    df = pd.read_csv(dataset_path)
                elif dataset_path.suffix == '.xlsx':
                    df = pd.read_excel(dataset_path)
                elif dataset_path.suffix == '.parquet':
                    df = pd.read_parquet(dataset_path)
                
                datasets[dataset_id] = {
                    'dataframe': df,
                    'path': str(dataset_path),
                    'name': dataset_path.name
                }
                logger.info(f"Loaded dataset {dataset_id}: {df.shape}")
        
        if not datasets:
            logger.error(f"No datasets found for report generation: {request.datasets}")
            return
        
        # Initialize comprehensive data processor and other engines
        from app.core.data_processor import ComprehensiveDataProcessor
        from app.core.ai_insights_engine import AIInsightsEngine
        from app.core.report_generator import ComprehensiveReportGenerator
        
        processor = ComprehensiveDataProcessor()
        insights_engine = AIInsightsEngine()
        report_generator = ComprehensiveReportGenerator()
        
        # Process each dataset
        dataset_analyses = {}
        for dataset_id, dataset_info in datasets.items():
            df = dataset_info['dataframe']
            
            # Perform comprehensive analysis
            analysis_results = processor.process_dataset(
                df=df,
                target_column=None,
                dataset_name=dataset_id
            )
            
            # Generate AI insights
            ai_insights = insights_engine.generate_comprehensive_insights(
                dataset_info=analysis_results['dataset_info'],
                profiling_results=analysis_results['profiling'],
                analysis_results=analysis_results,
                model_results=None
            )
            
            dataset_analyses[dataset_id] = {
                'analysis_results': analysis_results,
                'ai_insights': ai_insights,
                'dataset_info': dataset_info
            }
        
        # Generate comprehensive multi-dataset report
        comprehensive_report = report_generator.generate_multi_dataset_report(
            datasets_analysis=dataset_analyses,
            report_config={
                'title': request.title,
                'description': request.description,
                'sections': request.sections,
                'format': request.format.value,
                'include_visualizations': request.include_visualizations,
                'include_recommendations': request.include_recommendations,
                'custom_branding': request.custom_branding
            }
        )
        
        # Save report files
        reports_folder = data_folder / "reports" / report_id
        reports_folder.mkdir(parents=True, exist_ok=True)
        
        # Save HTML report
        html_path = None
        if comprehensive_report.get('html_content'):
            html_path = reports_folder / f"{report_id}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(comprehensive_report['html_content'])
            logger.info(f"HTML report saved: {html_path}")
        
        # Save PDF report if requested
        pdf_path = None
        if request.format in [ReportFormat.PDF, ReportFormat.BOTH] and comprehensive_report.get('pdf_content'):
            pdf_path = reports_folder / f"{report_id}.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(comprehensive_report['pdf_content'])
            logger.info(f"PDF report saved: {pdf_path}")
        
        # Save JSON data
        json_path = reports_folder / f"{report_id}_data.json"
        report_data = {
            'report_id': report_id,
            'title': request.title,
            'description': request.description,
            'datasets': list(datasets.keys()),
            'comprehensive_results': _make_report_serializable(comprehensive_report),
            'datasets_analysis': _make_report_serializable(dataset_analyses),
            'generation_config': {
                'sections': request.sections,
                'format': request.format.value,
                'include_visualizations': request.include_visualizations,
                'include_recommendations': request.include_recommendations
            },
            'execution_time': time.time() - start_time,
            'generated_at': time.time(),
            'status': 'completed'
        }
        
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Update report status
        status_data = {
            'report_id': report_id,
            'status': 'completed',
            'progress': 100,
            'execution_time': time.time() - start_time,
            'file_paths': {
                'html': str(html_path) if html_path else None,
                'pdf': str(pdf_path) if pdf_path else None,
                'data': str(json_path)
            },
            'file_sizes': {
                'html': html_path.stat().st_size if html_path and html_path.exists() else 0,
                'pdf': pdf_path.stat().st_size if pdf_path and pdf_path.exists() else 0,
                'data': json_path.stat().st_size if json_path.exists() else 0
            },
            'completed_at': time.time(),
            'insights_count': sum(len(analysis.get('ai_insights', {}).get('key_findings', [])) for analysis in dataset_analyses.values()),
            'visualizations_count': comprehensive_report.get('visualizations_count', 0)
        }
        
        status_path = reports_folder / "status.json"
        with open(status_path, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        execution_time = time.time() - start_time
        logger.info(f"Comprehensive report generation completed: {report_id} in {execution_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in comprehensive background report generation {report_id}: {e}")
        
        # Save error status
        try:
            error_folder = Path("data/reports") / report_id
            error_folder.mkdir(parents=True, exist_ok=True)
            
            error_status = {
                'report_id': report_id,
                'status': 'error',
                'error_message': str(e),
                'error_timestamp': time.time(),
                'progress': 0
            }
            
            error_file = error_folder / "status.json"
            with open(error_file, 'w') as f:
                json.dump(error_status, f, indent=2)
        except Exception as save_error:
            logger.error(f"Failed to save report error status: {save_error}")


def _make_report_serializable(obj):
    """Convert complex objects to JSON-serializable format for reports."""
    import numpy as np
    import pandas as pd
    from datetime import datetime
    
    if isinstance(obj, dict):
        return {key: _make_report_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_report_serializable(item) for item in obj]
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
        return _make_report_serializable(obj.to_dict())
    elif hasattr(obj, '__dict__'):
        return _make_report_serializable(obj.__dict__)
    else:
        return obj


async def generate_custom_report_content(report_id: str, request: CustomReportRequest):
    """Background task to generate custom report.
    
    Args:
        report_id: Unique report identifier
        request: Custom report configuration
    """
    try:
        import time
        import json
        from pathlib import Path
        import pandas as pd
        
        start_time = time.time()
        logger.info(f"Starting comprehensive custom report generation: {report_id}")
        
        # Find and load the dataset
        dataset_path = None
        data_folder = Path("data")
        
        for folder in ["uploads", "processed"]:
            folder_path = data_folder / folder
            if folder_path.exists():
                for file_path in folder_path.glob(f"*{request.dataset_id}*"):
                    if file_path.suffix in ['.csv', '.xlsx', '.parquet']:
                        dataset_path = file_path
                        break
        
        if not dataset_path:
            logger.error(f"Dataset not found for custom report: {request.dataset_id}")
            return
        
        # Load dataset
        if dataset_path.suffix == '.csv':
            df = pd.read_csv(dataset_path)
        elif dataset_path.suffix == '.xlsx':
            df = pd.read_excel(dataset_path)
        elif dataset_path.suffix == '.parquet':
            df = pd.read_parquet(dataset_path)
        
        logger.info(f"Dataset loaded for custom report: {df.shape}")
        
        # Initialize engines
        from app.core.data_processor import ComprehensiveDataProcessor
        from app.core.ai_insights_engine import AIInsightsEngine
        from app.core.report_generator import ComprehensiveReportGenerator
        
        processor = ComprehensiveDataProcessor()
        insights_engine = AIInsightsEngine()
        report_generator = ComprehensiveReportGenerator()
        
        # Perform analysis if needed
        analysis_results = None
        ai_insights = None
        
        if request.include_analysis:
            analysis_results = processor.process_dataset(
                df=df,
                target_column=None,
                dataset_name=request.dataset_id
            )
            
            ai_insights = insights_engine.generate_comprehensive_insights(
                dataset_info=analysis_results['dataset_info'],
                profiling_results=analysis_results['profiling'],
                analysis_results=analysis_results,
                model_results=None
            )
        
        # Generate custom visualizations if requested
        custom_visualizations = []
        if request.visualizations:
            for viz_config in request.visualizations:
                viz_result = processor.generate_custom_visualization(
                    df=df,
                    chart_type=viz_config.get('chart_type', 'bar'),
                    x_column=viz_config.get('x_column'),
                    y_column=viz_config.get('y_column'),
                    color_column=viz_config.get('color_column'),
                    title=viz_config.get('title'),
                    interactive=True,
                    save_path=str(data_folder / "charts" / f"custom_{report_id}_{len(custom_visualizations)}")
                )
                
                if viz_result.get('success'):
                    custom_visualizations.append(viz_result)
        
        # Generate custom report
        custom_report = report_generator.generate_custom_report(
            dataset_info={
                'dataframe': df,
                'name': request.dataset_id,
                'path': str(dataset_path)
            },
            analysis_results=analysis_results,
            ai_insights=ai_insights,
            custom_config={
                'title': request.title,
                'description': request.description,
                'sections': request.sections,
                'template': request.template,
                'custom_content': request.custom_content,
                'visualizations': custom_visualizations,
                'include_analysis': request.include_analysis,
                'custom_branding': request.custom_branding
            }
        )
        
        # Save custom report files
        reports_folder = data_folder / "reports" / report_id
        reports_folder.mkdir(parents=True, exist_ok=True)
        
        # Save HTML report
        html_path = None
        if custom_report.get('html_content'):
            html_path = reports_folder / f"{report_id}_custom.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(custom_report['html_content'])
            logger.info(f"Custom HTML report saved: {html_path}")
        
        # Save PDF if requested
        pdf_path = None
        if custom_report.get('pdf_content'):
            pdf_path = reports_folder / f"{report_id}_custom.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(custom_report['pdf_content'])
            logger.info(f"Custom PDF report saved: {pdf_path}")
        
        # Save custom report metadata
        execution_time = time.time() - start_time
        custom_metadata = {
            'report_id': report_id,
            'title': request.title,
            'description': request.description,
            'dataset_id': request.dataset_id,
            'dataset_path': str(dataset_path),
            'custom_config': {
                'sections': request.sections,
                'template': request.template,
                'include_analysis': request.include_analysis,
                'visualizations_count': len(custom_visualizations)
            },
            'analysis_included': analysis_results is not None,
            'insights_included': ai_insights is not None,
            'execution_time': execution_time,
            'generated_at': time.time(),
            'status': 'completed'
        }
        
        metadata_path = reports_folder / f"{report_id}_custom_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(_make_report_serializable(custom_metadata), f, indent=2, default=str)
        
        # Update status
        status_data = {
            'report_id': report_id,
            'type': 'custom',
            'status': 'completed',
            'progress': 100,
            'execution_time': execution_time,
            'file_paths': {
                'html': str(html_path) if html_path else None,
                'pdf': str(pdf_path) if pdf_path else None,
                'metadata': str(metadata_path)
            },
            'file_sizes': {
                'html': html_path.stat().st_size if html_path and html_path.exists() else 0,
                'pdf': pdf_path.stat().st_size if pdf_path and pdf_path.exists() else 0,
                'metadata': metadata_path.stat().st_size if metadata_path.exists() else 0
            },
            'completed_at': time.time(),
            'custom_sections_count': len(request.sections),
            'visualizations_count': len(custom_visualizations)
        }
        
        status_path = reports_folder / "status.json"
        with open(status_path, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        logger.info(f"Custom report generation completed: {report_id} in {execution_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in comprehensive custom report generation {report_id}: {e}")
        
        # Save error status
        try:
            error_folder = Path("data/reports") / report_id
            error_folder.mkdir(parents=True, exist_ok=True)
            
            error_status = {
                'report_id': report_id,
                'type': 'custom',
                'status': 'error',
                'error_message': str(e),
                'error_timestamp': time.time(),
                'progress': 0
            }
            
            error_file = error_folder / "status.json"
            with open(error_file, 'w') as f:
                json.dump(error_status, f, indent=2)
        except Exception as save_error:
            logger.error(f"Failed to save custom report error status: {save_error}")
