"""
EchoAI Gateway - Main Application Entry Point

This is the unified gateway for the EchoAI platform. All services are registered
as FastAPI routers and mounted under a single application.

Features:
- Database connection pool (PostgreSQL with asyncpg)
- Memcached session caching (optional, configurable)
- Auth middleware (optional vs required mode via AUTH_ENFORCEMENT)
- CORS configuration for development
- Tool operation logging middleware

Lifespan Events:
- Startup: Initialize database pool, connect to Memcached
- Shutdown: Close database connections, disconnect from Memcached

Configuration (via environment/.env):
- DATABASE_URL: PostgreSQL connection string
- MEMCACHED_ENABLED: Enable/disable Memcached caching
- MEMCACHED_HOSTS: Memcached server addresses
- AUTH_ENFORCEMENT: "optional" or "required"
"""

from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from echolib.config import settings
from echolib.database import database_lifespan
from echolib.cache import cache_lifespan
from echolib.security import AuthMiddleware

# Import containers to register providers
from apps.session import container as _sess_container  # noqa: F401
from apps.rag import container as _rag_container  # noqa: F401
from apps.llm import container as _llm_container  # noqa: F401
from apps.tool import container as _tool_container  # noqa: F401
from apps.appmgr import container as _app_container  # noqa: F401
from apps.workflow import container as _wf_container  # noqa: F401
from apps.agent import container as _agent_container  # noqa: F401
from apps.connector import container as _conn_container  # noqa: F401
from apps.chat import container as _chat_container  # noqa: F401
from apps.workbench import container as _wb_container  # noqa: F401

# Routers
from apps.session.routes import router as session_router
from apps.rag.routes import router as rag_router
from apps.llm.routes import router as llm_router
from apps.tool.routes import router as tool_router
from apps.appmgr.routes import router as app_router
from apps.appmgr.routes import lookup_router as app_lookup_router
from apps.appmgr.routes import skills_router as app_skills_router
from apps.workflow.routes import router as wf_router
from apps.agent.routes import router as agent_router
from apps.connector.routes import router as conn_router
from apps.chat.routes import router as chat_router
from apps.workbench.routes import router as workbench_router


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown of:
    - Database connection pool
    - Memcached cache client
    """
    logger.info("=" * 60)
    logger.info("🚀 Starting EchoAI Gateway...")
    logger.info("=" * 60)
    logger.info(f"📋 Configuration:")
    logger.info(f"   - Auth Enforcement: {settings.auth_enforcement}")
    logger.info(f"   - Memcached Enabled: {settings.memcached_enabled}")
    logger.info(f"   - Service Mode: {settings.service_mode}")

    # Initialize database
    async with database_lifespan():
        logger.info("✅ Database connection pool initialized")

        # Initialize cache (optional - gracefully handles connection failures)
        async with cache_lifespan():
            if settings.memcached_enabled:
                from echolib.cache import cache_client
                if cache_client.is_available:
                    logger.info("✅ Memcached cache layer initialized")
                else:
                    logger.warning("⚠️ Memcached unavailable - using database only")
            else:
                logger.info("ℹ️ Memcached disabled - using database only")

            logger.info("=" * 60)
            logger.info("🎉 EchoAI Gateway READY")
            logger.info("=" * 60)
            yield

    logger.info("👋 EchoAI Gateway shutdown complete")


# Create FastAPI application with lifespan
app = FastAPI(
    title=f"{settings.app_name} (Gateway)",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ==============================================================================
# Middleware Configuration
# ==============================================================================

# CORS middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth middleware - extracts user from JWT and attaches to request.state
app.add_middleware(AuthMiddleware)

# Mount static directory to serve uploaded logos (e.g., /static/logos/<file>.png)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Tool System Logging Middleware
@app.middleware("http")
async def log_tool_operations(request: Request, call_next):
    """
    Middleware to log all tool-related operations for debugging and monitoring.
    Logs: discovery, invocation, registration, deletion, and errors.
    """
    start_time = time.time()
    path = request.url.path
    method = request.method

    # Check if this is a tool-related request
    is_tool_request = path.startswith("/tools")

    if is_tool_request:
        logger.info(f"Tool Request: {method} {path}")

        # Log request details for specific operations
        if "/tools/discover" in path:
            logger.info("Tool Discovery initiated")
        elif "/tools/invoke" in path:
            tool_identifier = path.split("/")[-1] if "/" in path else "unknown"
            logger.info(f"Tool Invocation: {tool_identifier}")
        elif "/tools/list" in path:
            logger.info("Listing all registered tools")
        elif method == "POST" and path == "/tools/register":
            logger.info("Tool Registration request")
        elif method == "DELETE":
            tool_id = path.split("/")[-1]
            logger.info(f"Tool Deletion request: {tool_id}")

    # Process request
    try:
        response = await call_next(request)

        # Log completion with execution time
        if is_tool_request:
            duration = (time.time() - start_time) * 1000  # Convert to ms
            status = response.status_code

            if status >= 200 and status < 300:
                logger.info(
                    f"Tool Request completed: {method} {path} - "
                    f"Status: {status} - Duration: {duration:.2f}ms"
                )
            elif status >= 400 and status < 500:
                logger.warning(
                    f"Tool Request client error: {method} {path} - "
                    f"Status: {status} - Duration: {duration:.2f}ms"
                )
            elif status >= 500:
                logger.error(
                    f"Tool Request server error: {method} {path} - "
                    f"Status: {status} - Duration: {duration:.2f}ms"
                )

        return response

    except Exception as e:
        duration = (time.time() - start_time) * 1000
        logger.error(
            f"Tool Request exception: {method} {path} - "
            f"Error: {str(e)} - Duration: {duration:.2f}ms"
        )
        raise


# ==============================================================================
# Health Check
# ==============================================================================


@app.get("/healthz")
async def healthz():
    """
    Health check endpoint.

    Returns:
        Status and service mode.
    """
    return {
        "status": "ok",
        "mode": settings.service_mode,
        "auth_enforcement": settings.auth_enforcement,
    }


@app.get("/health/db")
async def health_db():
    """
    Database health check.

    Tests database connectivity by executing a simple query.
    """
    from echolib.database import get_db_session
    from sqlalchemy import text

    try:
        async with get_db_session() as db:
            result = await db.execute(text("SELECT 1"))
            _ = result.scalar()
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return {"status": "error", "database": str(e)}


@app.get("/health/cache")
async def health_cache():
    """
    Cache health check.

    Tests Memcached connectivity by performing a test set/get operation.
    """
    from echolib.cache import cache_client

    response = {
        "status": "ok" if cache_client.is_available else "unavailable",
        "cache_enabled": settings.memcached_enabled,
        "cache_available": cache_client.is_available,
        "hosts": settings.memcached_hosts if settings.memcached_enabled else None,
        "ttl_seconds": settings.memcached_ttl if settings.memcached_enabled else None,
    }

    # If cache is available, test it with a ping
    if cache_client.is_available:
        test_key = "health_check_test"
        test_value = {"test": "ping", "timestamp": time.time()}
        set_result = await cache_client.set(test_key, test_value, ttl=10)
        get_result = await cache_client.get(test_key)
        await cache_client.delete(test_key)

        response["test_set"] = "ok" if set_result else "failed"
        response["test_get"] = "ok" if get_result else "failed"

    return response


# ==============================================================================
# Router Registration
# ==============================================================================

app.include_router(session_router)
app.include_router(rag_router)
app.include_router(llm_router)
app.include_router(tool_router)
app.include_router(app_router)
app.include_router(app_lookup_router)
app.include_router(app_skills_router)
app.include_router(wf_router)
app.include_router(agent_router)
app.include_router(conn_router)
app.include_router(chat_router)
app.include_router(workbench_router, prefix="/api")


# ==============================================================================
# WebSocket Endpoint for Execution Transparency
# ==============================================================================


@app.websocket("/ws/execution/{run_id}")
async def websocket_execution(websocket: WebSocket, run_id: str):
    """
    WebSocket endpoint for real-time execution transparency.

    Clients connect to receive step-level events during workflow execution.
    Events include: run_started, step_started, step_completed, step_failed,
    run_completed, run_failed.

    Args:
        websocket: WebSocket connection
        run_id: Run identifier to subscribe to

    Usage:
        1. POST /workflows/chat/send returns run_id immediately
        2. Client connects to ws://host/ws/execution/{run_id}
        3. Client receives events as workflow executes
        4. Connection closes after run_completed/run_failed
    """
    from apps.workflow.runtime.ws_manager import ws_manager

    await ws_manager.connect(run_id, websocket)
    try:
        # Keep-alive loop - wait for messages or disconnect
        while True:
            # receive_text() will raise WebSocketDisconnect on client disconnect
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(run_id, websocket)
        logger.info(f"WebSocket client disconnected from run {run_id}")
