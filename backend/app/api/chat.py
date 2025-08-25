"""AI Chat Interface API endpoints."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = datetime.now()
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str
    session_id: Optional[str] = None
    dataset_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    session_id: str
    message_id: str
    suggestions: List[str] = []
    data_results: Optional[Dict[str, Any]] = None
    visualizations: List[Dict[str, Any]] = []
    timestamp: datetime = datetime.now()


class DataQueryRequest(BaseModel):
    """Data query through natural language."""
    query: str
    dataset_id: str
    session_id: Optional[str] = None


class DataQueryResponse(BaseModel):
    """Data query response."""
    query: str
    sql_generated: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    explanation: str
    visualizations: List[Dict[str, Any]] = []
    session_id: str


@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """Chat with AI assistant about data analysis.
    
    Args:
        request: Chat request with message and context
        
    Returns:
        AI response with suggestions and data insights
    """
    try:
        logger.info(f"Processing chat request: {request.message[:100]}...")
        
        # Generate session ID if not provided
        session_id = request.session_id or f"chat_{int(time.time())}"
        message_id = f"msg_{int(time.time())}"
        
        # Initialize AI insights engine for conversational analysis
        from app.core.ai_insights_engine import AIInsightsEngine
        insights_engine = AIInsightsEngine()
        
        # Load dataset if specified
        dataset_context = None
        if request.dataset_id:
            dataset_context = await _load_dataset_context(request.dataset_id)
        
        # Process the chat message through AI engine
        chat_response = insights_engine.process_chat_message(
            message=request.message,
            session_id=session_id,
            dataset_context=dataset_context,
            user_context=request.context or {}
        )
        
        # Extract data results if the query was data-related
        data_results = None
        visualizations = []
        
        if chat_response.get('data_query_detected') and dataset_context:
            # Execute data query
            query_results = await _execute_data_query(
                query=request.message,
                dataset_context=dataset_context
            )
            
            data_results = query_results.get('results')
            visualizations = query_results.get('visualizations', [])
        
        # Generate contextual suggestions
        suggestions = _generate_chat_suggestions(
            message=request.message,
            response=chat_response.get('response', ''),
            dataset_available=dataset_context is not None
        )
        
        # Save chat history
        await _save_chat_message(
            session_id=session_id,
            user_message=request.message,
            ai_response=chat_response.get('response', ''),
            metadata={
                'dataset_id': request.dataset_id,
                'data_query_detected': chat_response.get('data_query_detected', False),
                'suggestions': suggestions
            }
        )
        
        return ChatResponse(
            response=chat_response.get('response', 'I understand your question. How can I help you analyze your data?'),
            session_id=session_id,
            message_id=message_id,
            suggestions=suggestions,
            data_results=data_results,
            visualizations=visualizations
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Error processing chat request")


@router.post("/chat/query-data", response_model=DataQueryResponse)
async def query_data_with_natural_language(request: DataQueryRequest):
    """Query dataset using natural language.
    
    Args:
        request: Natural language data query request
        
    Returns:
        Query results with explanation and visualizations
    """
    try:
        logger.info(f"Processing natural language data query: {request.query}")
        
        # Generate session ID if not provided
        session_id = request.session_id or f"query_{int(time.time())}"
        
        # Load dataset
        dataset_context = await _load_dataset_context(request.dataset_id)
        if not dataset_context:
            raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not found")
        
        # Initialize AI insights engine for NL to SQL conversion
        from app.core.ai_insights_engine import AIInsightsEngine
        insights_engine = AIInsightsEngine()
        
        # Process natural language query
        query_analysis = insights_engine.process_natural_language_query(
            query=request.query,
            dataset_info=dataset_context['dataset_info'],
            column_info=dataset_context['column_info']
        )
        
        # Execute the query
        query_results = await _execute_advanced_data_query(
            query=request.query,
            query_analysis=query_analysis,
            dataset_context=dataset_context
        )
        
        # Generate visualizations if appropriate
        visualizations = []
        if query_results.get('visualization_suggested'):
            viz_results = await _generate_query_visualizations(
                query=request.query,
                results=query_results.get('results'),
                dataset_context=dataset_context
            )
            visualizations = viz_results.get('visualizations', [])
        
        return DataQueryResponse(
            query=request.query,
            sql_generated=query_analysis.get('sql_query'),
            results=query_results.get('results'),
            explanation=query_analysis.get('explanation', 'Query executed successfully'),
            visualizations=visualizations,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Error processing natural language query: {e}")
        raise HTTPException(status_code=500, detail="Error processing natural language query")


@router.get("/chat/sessions/{session_id}/history")
async def get_chat_history(session_id: str):
    """Get chat history for a session.
    
    Args:
        session_id: Chat session identifier
        
    Returns:
        Chat history with messages
    """
    try:
        chat_folder = Path("data/chat_sessions") / session_id
        history_file = chat_folder / "history.json"
        
        if not history_file.exists():
            return {"session_id": session_id, "messages": [], "total_messages": 0}
        
        with open(history_file, 'r') as f:
            history_data = json.load(f)
        
        return {
            "session_id": session_id,
            "messages": history_data.get('messages', []),
            "total_messages": len(history_data.get('messages', [])),
            "created_at": history_data.get('created_at'),
            "last_updated": history_data.get('last_updated')
        }
        
    except Exception as e:
        logger.error(f"Error retrieving chat history for {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving chat history")


@router.get("/chat/sessions")
async def list_chat_sessions(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page")
):
    """List all chat sessions.
    
    Args:
        page: Page number
        page_size: Items per page
        
    Returns:
        List of chat sessions
    """
    try:
        chat_folder = Path("data/chat_sessions")
        if not chat_folder.exists():
            return {"sessions": [], "total_sessions": 0, "page": page, "page_size": page_size}
        
        sessions = []
        for session_folder in chat_folder.iterdir():
            if session_folder.is_dir():
                history_file = session_folder / "history.json"
                if history_file.exists():
                    with open(history_file, 'r') as f:
                        history_data = json.load(f)
                    
                    sessions.append({
                        "session_id": session_folder.name,
                        "message_count": len(history_data.get('messages', [])),
                        "created_at": history_data.get('created_at'),
                        "last_updated": history_data.get('last_updated'),
                        "dataset_ids": list(set([
                            msg.get('metadata', {}).get('dataset_id') 
                            for msg in history_data.get('messages', [])
                            if msg.get('metadata', {}).get('dataset_id')
                        ]))
                    })
        
        # Sort by last updated
        sessions.sort(key=lambda x: x.get('last_updated', ''), reverse=True)
        
        # Pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_sessions = sessions[start_idx:end_idx]
        
        return {
            "sessions": paginated_sessions,
            "total_sessions": len(sessions),
            "page": page,
            "page_size": page_size
        }
        
    except Exception as e:
        logger.error(f"Error listing chat sessions: {e}")
        raise HTTPException(status_code=500, detail="Error listing chat sessions")


@router.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session and its history.
    
    Args:
        session_id: Chat session identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        import shutil
        
        chat_folder = Path("data/chat_sessions") / session_id
        
        if chat_folder.exists():
            shutil.rmtree(chat_folder)
            logger.info(f"Chat session deleted: {session_id}")
        
        return {
            "session_id": session_id,
            "deleted": True,
            "message": "Chat session deleted successfully",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error deleting chat session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Error deleting chat session")


async def _load_dataset_context(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Load dataset context for chat analysis."""
    try:
        data_folder = Path("data")
        
        # Find dataset file
        dataset_path = None
        for folder in ["uploads", "processed"]:
            folder_path = data_folder / folder
            if folder_path.exists():
                for file_path in folder_path.glob(f"*{dataset_id}*"):
                    if file_path.suffix in ['.csv', '.xlsx', '.parquet']:
                        dataset_path = file_path
                        break
        
        if not dataset_path:
            return None
        
        # Load dataset
        if dataset_path.suffix == '.csv':
            df = pd.read_csv(dataset_path)
        elif dataset_path.suffix == '.xlsx':
            df = pd.read_excel(dataset_path)
        elif dataset_path.suffix == '.parquet':
            df = pd.read_parquet(dataset_path)
        
        # Generate dataset context
        return {
            'dataset_id': dataset_id,
            'dataframe': df,
            'dataset_path': str(dataset_path),
            'dataset_info': {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'memory_usage': df.memory_usage(deep=True).sum()
            },
            'column_info': {
                col: {
                    'dtype': str(df[col].dtype),
                    'null_count': df[col].isnull().sum(),
                    'unique_count': df[col].nunique(),
                    'sample_values': df[col].dropna().head(5).tolist()
                }
                for col in df.columns
            }
        }
        
    except Exception as e:
        logger.error(f"Error loading dataset context for {dataset_id}: {e}")
        return None


async def _execute_data_query(query: str, dataset_context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute data query based on natural language."""
    try:
        from app.core.data_processor import ComprehensiveDataProcessor
        
        processor = ComprehensiveDataProcessor()
        df = dataset_context['dataframe']
        
        # Simple query processing for demonstration
        query_lower = query.lower()
        results = {}
        visualizations = []
        
        if 'shape' in query_lower or 'size' in query_lower:
            results = {
                'type': 'dataset_info',
                'shape': df.shape,
                'rows': df.shape[0],
                'columns': df.shape[1]
            }
        elif 'columns' in query_lower:
            results = {
                'type': 'columns_info',
                'columns': df.columns.tolist(),
                'total_columns': len(df.columns)
            }
        elif 'summary' in query_lower or 'describe' in query_lower:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                results = {
                    'type': 'summary_statistics',
                    'summary': df[numeric_cols].describe().to_dict()
                }
        
        return {
            'results': results,
            'visualizations': visualizations
        }
        
    except Exception as e:
        logger.error(f"Error executing data query: {e}")
        return {'results': None, 'visualizations': []}


async def _execute_advanced_data_query(
    query: str, 
    query_analysis: Dict[str, Any], 
    dataset_context: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute advanced data query with AI analysis."""
    try:
        df = dataset_context['dataframe']
        
        # Execute based on query analysis
        query_type = query_analysis.get('query_type', 'unknown')
        
        if query_type == 'aggregation':
            # Handle aggregation queries
            return await _handle_aggregation_query(query_analysis, df)
        elif query_type == 'filtering':
            # Handle filtering queries
            return await _handle_filtering_query(query_analysis, df)
        elif query_type == 'statistical':
            # Handle statistical queries
            return await _handle_statistical_query(query_analysis, df)
        else:
            # Default handling
            return {'results': {'message': 'Query processed'}, 'visualization_suggested': False}
        
    except Exception as e:
        logger.error(f"Error executing advanced data query: {e}")
        return {'results': None, 'visualization_suggested': False}


async def _handle_aggregation_query(query_analysis: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Handle aggregation queries."""
    try:
        # Simple aggregation example
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            results = {
                'type': 'aggregation',
                'aggregations': {
                    col: {
                        'sum': df[col].sum(),
                        'mean': df[col].mean(),
                        'count': df[col].count()
                    }
                    for col in numeric_cols[:5]  # Limit to first 5 numeric columns
                }
            }
            return {'results': results, 'visualization_suggested': True}
        
        return {'results': {'message': 'No numeric columns found for aggregation'}, 'visualization_suggested': False}
        
    except Exception as e:
        logger.error(f"Error handling aggregation query: {e}")
        return {'results': None, 'visualization_suggested': False}


async def _handle_filtering_query(query_analysis: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Handle filtering queries."""
    try:
        # Simple filtering example - return first 10 rows
        results = {
            'type': 'filtered_data',
            'sample_data': df.head(10).to_dict('records'),
            'total_rows': len(df)
        }
        return {'results': results, 'visualization_suggested': False}
        
    except Exception as e:
        logger.error(f"Error handling filtering query: {e}")
        return {'results': None, 'visualization_suggested': False}


async def _handle_statistical_query(query_analysis: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Handle statistical queries."""
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            results = {
                'type': 'statistical_analysis',
                'statistics': df[numeric_cols].describe().to_dict(),
                'correlations': df[numeric_cols].corr().to_dict() if len(numeric_cols) > 1 else {}
            }
            return {'results': results, 'visualization_suggested': True}
        
        return {'results': {'message': 'No numeric columns found for statistical analysis'}, 'visualization_suggested': False}
        
    except Exception as e:
        logger.error(f"Error handling statistical query: {e}")
        return {'results': None, 'visualization_suggested': False}


async def _generate_query_visualizations(
    query: str, 
    results: Dict[str, Any], 
    dataset_context: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate visualizations for query results."""
    try:
        from app.core.data_processor import ComprehensiveDataProcessor
        
        processor = ComprehensiveDataProcessor()
        df = dataset_context['dataframe']
        
        # Generate appropriate visualizations based on results type
        visualizations = []
        
        if results and results.get('type') == 'aggregation':
            # Create bar chart for aggregations
            viz_result = processor.generate_custom_visualization(
                df=df,
                chart_type='bar',
                title='Aggregation Results',
                interactive=True,
                save_path=str(Path("data/charts") / f"query_viz_{int(time.time())}")
            )
            
            if viz_result.get('success'):
                visualizations.append(viz_result)
        
        return {'visualizations': visualizations}
        
    except Exception as e:
        logger.error(f"Error generating query visualizations: {e}")
        return {'visualizations': []}


def _generate_chat_suggestions(message: str, response: str, dataset_available: bool) -> List[str]:
    """Generate contextual chat suggestions."""
    suggestions = []
    
    if dataset_available:
        suggestions.extend([
            "Show me the dataset summary",
            "What are the column names?",
            "Generate visualizations for this data",
            "Find correlations in the data",
            "Detect outliers in the dataset"
        ])
    else:
        suggestions.extend([
            "Upload a dataset to analyze",
            "Show me example analyses",
            "What can this platform do?",
            "How do I get started?"
        ])
    
    # Add contextual suggestions based on message content
    message_lower = message.lower()
    if 'visualization' in message_lower or 'chart' in message_lower:
        suggestions.append("Generate automatic visualizations")
    if 'model' in message_lower or 'prediction' in message_lower:
        suggestions.append("Train a machine learning model")
    if 'report' in message_lower:
        suggestions.append("Generate a comprehensive report")
    
    return suggestions[:5]  # Limit to 5 suggestions


async def _save_chat_message(
    session_id: str, 
    user_message: str, 
    ai_response: str, 
    metadata: Dict[str, Any]
):
    """Save chat message to session history."""
    try:
        chat_folder = Path("data/chat_sessions") / session_id
        chat_folder.mkdir(parents=True, exist_ok=True)
        
        history_file = chat_folder / "history.json"
        
        # Load existing history
        if history_file.exists():
            with open(history_file, 'r') as f:
                history_data = json.load(f)
        else:
            history_data = {
                'session_id': session_id,
                'created_at': datetime.now().isoformat(),
                'messages': []
            }
        
        # Add new messages
        timestamp = datetime.now().isoformat()
        
        history_data['messages'].extend([
            {
                'role': 'user',
                'content': user_message,
                'timestamp': timestamp,
                'metadata': metadata
            },
            {
                'role': 'assistant',
                'content': ai_response,
                'timestamp': timestamp,
                'metadata': metadata
            }
        ])
        
        history_data['last_updated'] = timestamp
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Error saving chat message: {e}")
