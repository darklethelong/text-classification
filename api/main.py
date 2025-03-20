"""
Complaint Detection API
-----------------------
This module provides the main FastAPI application entry point for the complaint detection service.
"""

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from config import settings
from routers import auth, complaints

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set up FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="API for detecting complaints in customer service conversations",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(complaints.router)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI documentation."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - API Documentation",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3/swagger-ui.css",
    )


@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    """OpenAPI schema endpoint."""
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )


if __name__ == "__main__":
    logger.info(f"Starting {settings.app_name} on http://0.0.0.0:8000")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Documentation available at http://0.0.0.0:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug) 