"""
Script to run the FastAPI application for complaint detection.
"""

import uvicorn

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True) 