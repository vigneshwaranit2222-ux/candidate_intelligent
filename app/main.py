"""
FastAPI Application Entry Point & Lifespan Controller (`main.py`).

Analogy for Beginners:
Think of `main.py` like opening the front doors of a high-tech university campus!
When the server turns on (Lifespan Startup), it prepares classroom tables (database schemas) and turns on lights (router endpoints).
Middlewares act like friendly security guards at the door: they verify requests, prevent unexpected crashes,
and ensure cross-origin browsers (CORS) can safely communicate with our API!
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.db.session import async_engine
from app.db.models import Base
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan Context Manager.
    Handles startup table creation and graceful shutdown resource cleanup.
    """
    # Startup: Create Database Tables automatically if they do not exist
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield  # Application runs and handles HTTP traffic

    # Shutdown: Dispose database connections cleanly
    await async_engine.dispose()


# Initialize FastAPI Application Singleton Instance
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Phase 1 AI-Powered Candidate Intelligence and Job Readiness Platform API",
    lifespan=lifespan
)


# Configure Cross-Origin Resource Sharing (CORS) Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any frontend origin in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handler Middleware
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches all uncaught exceptions across the app to guarantee zero unhandled runtime crashes.
    Returns clean JSON error responses instead of HTML tracebacks.
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "path": request.url.path
        }
    )


# Register API Routes
app.include_router(router)


# Root Health Check Endpoint
@app.get("/", tags=["Health Check"])
async def root_health_check():
    """Returns application name, operational status, and current version."""
    return {
        "status": "online",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "message": "AI Candidate Intelligence Platform Phase 1 is running smoothly!"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
