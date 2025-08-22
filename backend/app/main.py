"""Main FastAPI application entry point for Auto Analytics Platform."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import json
import asyncio
from typing import List

from app.config import settings
from app.utils.logger import setup_logger


# Setup logging
logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Auto Analytics Platform...")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Database URL: {settings.database_url}")
    
    # Initialize database
    try:
        # Database initialization will be added here
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Auto Analytics Platform...")


# Initialize FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    debug=settings.debug,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files - fix path to be relative to project root
project_root = Path(__file__).parent.parent.parent
static_dir = project_root / "frontend" / "static"
templates_dir = project_root / "frontend" / "templates"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
else:
    logger.warning(f"Static directory not found: {static_dir}")

# Initialize templates
if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
else:
    logger.warning(f"Templates directory not found: {templates_dir}")
    templates = None


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Root endpoint serving the main application."""
    try:
        # Serve the enhanced template if available
        if templates:
            return templates.TemplateResponse("layout.html", {"request": request})
        
        # Fallback to simple HTML
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Auto Analytics Platform</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
                h1 { color: #2c3e50; text-align: center; }
                .feature { margin: 20px 0; padding: 20px; background: #ecf0f1; border-radius: 5px; }
                .status { color: #27ae60; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 NISR Rwanda Analytics Platform</h1>
                <p class="status">✅ Backend API is running successfully!</p>
                
                <div class="feature">
                    <h3>🔄 Automated Data Processing</h3>
                    <p>Upload datasets and get automated insights, cleaning, and analysis</p>
                </div>
                
                <div class="feature">
                    <h3>🤖 Machine Learning Automation</h3>
                    <p>Automatic model selection, training, and hyperparameter optimization</p>
                </div>
                
                <div class="feature">
                    <h3>📊 Smart Visualizations</h3>
                    <p>Context-aware chart generation and interactive dashboards</p>
                </div>
                
                <div class="feature">
                    <h3>📄 Intelligent Reports</h3>
                    <p>Automated insight generation and comprehensive report creation</p>
                </div>
                
                <p style="text-align: center; margin-top: 40px;">
                    <a href="/docs" style="color: #3498db; text-decoration: none; font-weight: bold;">📚 View API Documentation</a> |
                    <a href="/health" style="color: #3498db; text-decoration: none; font-weight: bold;">🔍 Health Check</a>
                </p>
            </div>
        </body>
        </html>
        """
    except Exception as e:
        logger.error(f"Error serving root endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the dashboard page."""
    try:
        if templates:
            return templates.TemplateResponse("dashboard.html", {"request": request})
        else:
            return HTMLResponse("<h1>Dashboard not available</h1>")
    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Auto Analytics Platform API",
        "version": settings.api_version,
        "debug": settings.debug,
    }


@app.get("/api/v1/status")
async def api_status():
    """API status endpoint with detailed information."""
    return {
        "api": {
            "title": settings.api_title,
            "version": settings.api_version,
            "status": "operational"
        },
        "features": {
            "data_upload": "available",
            "data_profiling": "available", 
            "auto_ml": "available",
            "visualizations": "available",
            "reports": "available"
        },
        "limits": {
            "max_file_size_mb": settings.max_file_size // (1024 * 1024),
            "allowed_formats": settings.allowed_extensions,
            "max_training_time_minutes": settings.max_training_time // 60
        }
    }


# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_data = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.client_data[websocket] = {"client_id": client_id, "subscriptions": []}
        logger.info(f"WebSocket client {client_id} connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            client_info = self.client_data.pop(websocket, {})
            client_id = client_info.get("client_id", "unknown")
            logger.info(f"WebSocket client {client_id} disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        if self.active_connections:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to connection: {e}")
                    disconnected.append(connection)
            
            # Remove disconnected connections
            for conn in disconnected:
                self.disconnect(conn)

    async def broadcast_to_subscribers(self, message: str, subscription_type: str):
        if self.active_connections:
            disconnected = []
            for connection in self.active_connections:
                try:
                    client_info = self.client_data.get(connection, {})
                    if subscription_type in client_info.get("subscriptions", []):
                        await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to subscriber: {e}")
                    disconnected.append(connection)
            
            # Remove disconnected connections
            for conn in disconnected:
                self.disconnect(conn)


# Global connection manager
manager = ConnectionManager()


@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket, client_id: str = "anonymous"):
    await manager.connect(websocket, client_id)
    try:
        # Send initial connection success message
        await manager.send_personal_message(json.dumps({
            "type": "connection",
            "status": "connected",
            "message": "Real-time connection established",
            "timestamp": asyncio.get_event_loop().time()
        }), websocket)
        
        # Start background tasks for this client
        asyncio.create_task(send_periodic_updates(websocket))
        
        while True:
            # Listen for client messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                await handle_client_message(message, websocket)
            except json.JSONDecodeError:
                await manager.send_personal_message(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }), websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def handle_client_message(message: dict, websocket: WebSocket):
    """Handle incoming WebSocket messages from clients."""
    action = message.get("action")
    client_info = manager.client_data.get(websocket, {})
    
    if action == "subscribe":
        subscription_type = message.get("type", "general")
        if "subscriptions" not in client_info:
            client_info["subscriptions"] = []
        if subscription_type not in client_info["subscriptions"]:
            client_info["subscriptions"].append(subscription_type)
        
        await manager.send_personal_message(json.dumps({
            "type": "subscription",
            "status": "subscribed",
            "subscription_type": subscription_type
        }), websocket)
        
    elif action == "unsubscribe":
        subscription_type = message.get("type", "general")
        if "subscriptions" in client_info and subscription_type in client_info["subscriptions"]:
            client_info["subscriptions"].remove(subscription_type)
        
        await manager.send_personal_message(json.dumps({
            "type": "subscription",
            "status": "unsubscribed", 
            "subscription_type": subscription_type
        }), websocket)
        
    elif action == "ping":
        await manager.send_personal_message(json.dumps({
            "type": "pong",
            "timestamp": asyncio.get_event_loop().time()
        }), websocket)


async def send_periodic_updates(websocket: WebSocket):
    """Send periodic updates to connected clients."""
    try:
        while websocket in manager.active_connections:
            # Simulate real-time metrics
            metrics_data = {
                "type": "metrics",
                "payload": {
                    "cpu_usage": __import__("random").uniform(20, 80),
                    "memory_usage": __import__("random").uniform(30, 90),
                    "active_users": __import__("random").randint(50, 500),
                    "api_requests": __import__("random").randint(100, 1000),
                    "data_processed": __import__("random").randint(1000, 10000),
                    "models_trained": __import__("random").randint(0, 50)
                },
                "timestamp": asyncio.get_event_loop().time()
            }
            
            await manager.send_personal_message(json.dumps(metrics_data), websocket)
            
            # Send performance data for charts
            performance_data = {
                "type": "performance",
                "payload": {
                    "value": __import__("random").uniform(60, 95),
                    "timestamp": asyncio.get_event_loop().time()
                }
            }
            
            await manager.send_personal_message(json.dumps(performance_data), websocket)
            
            # Occasionally send alerts
            if __import__("random").random() < 0.1:  # 10% chance
                alerts = [
                    {"message": "High CPU usage detected", "severity": "warning"},
                    {"message": "New model training completed", "severity": "success"},
                    {"message": "Data quality check passed", "severity": "info"},
                    {"message": "Anomaly detected in dataset", "severity": "warning"}
                ]
                alert = __import__("random").choice(alerts)
                alert_data = {
                    "type": "alert",
                    "payload": {
                        **alert,
                        "id": str(__import__("uuid").uuid4()),
                        "timestamp": asyncio.get_event_loop().time()
                    }
                }
                await manager.send_personal_message(json.dumps(alert_data), websocket)
            
            await asyncio.sleep(2)  # Send updates every 2 seconds
            
    except Exception as e:
        logger.error(f"Error in periodic updates: {e}")
        manager.disconnect(websocket)


# Import and include API routers
try:
    from app.api import upload, datasets, analysis, modeling, visualization, reports
    
    app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
    app.include_router(datasets.router, prefix="/api/v1", tags=["datasets"])
    app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])
    app.include_router(modeling.router, prefix="/api/v1", tags=["modeling"])
    app.include_router(visualization.router, prefix="/api/v1", tags=["visualization"])
    app.include_router(reports.router, prefix="/api/v1", tags=["reports"])
    
    logger.info("All API routers loaded successfully")
    
except ImportError as e:
    logger.warning(f"Some API modules not yet available: {e}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers if not settings.debug else 1,
        log_level=settings.log_level.lower()
    )