"""
FastAPI application for complaint detection.
Provides API endpoints for real-time complaint detection and visualization.
"""

import os
import time
import base64
import io
from typing import List, Dict, Any, Optional
from datetime import datetime

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException, Request, WebSocket, BackgroundTasks, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn
from io import BytesIO
import json
from collections import deque

import sys
sys.path.append('.')

from inference.inference import ComplaintDetector
from visualization.visualization import (
    plot_complaint_timeline, 
    plot_complaint_percentage_gauge,
    plot_complaint_heatmap, 
    plot_complaint_distribution
)

# Initialize FastAPI app
app = FastAPI(
    title="Complaint Detection API",
    description="API for real-time complaint detection and visualization",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory for visualization output
os.makedirs("api/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Create templates directory
templates = Jinja2Templates(directory="api/templates")

# Store active sessions and their data
sessions = {}
# Global model instance for all sessions to use
global_model = None

# Cache for plot images
visualization_cache = {}

# Input models
class TextInput(BaseModel):
    text: str = Field(..., description="Text to analyze for complaints")
    session_id: Optional[str] = Field(None, description="Session ID for tracking conversation")

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze for complaints")
    session_id: Optional[str] = Field(None, description="Session ID for tracking conversation")

class LiveSessionInput(BaseModel):
    session_name: str = Field(..., description="Name of the session")
    threshold: float = Field(0.3, description="Threshold for complaint classification")
    chunk_size: int = Field(4, description="Number of messages to process at once")

# Response models
class TextResponse(BaseModel):
    is_complaint: bool
    complaint_probability: float
    text: str
    timestamp: str
    complaint_intensity: str

class BatchResponse(BaseModel):
    results: List[TextResponse]
    complaint_percentage: float
    visualizations: Dict[str, str] = Field(
        {},
        description="Base64 encoded visualizations"
    )

class SessionInfo(BaseModel):
    session_id: str
    name: str
    created_at: str
    message_count: int
    complaint_count: int
    complaint_percentage: float

# Helper functions
def get_global_model():
    """Get or initialize the global model instance."""
    global global_model
    if global_model is None:
        try:
            # Initialize model with best available model
            latest_model_path = "models/mlm_full.pt"
            global_model = ComplaintDetector(
                model_type="mlm_full",
                model_path=latest_model_path,
                threshold=0.3
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    return global_model

def clear_visualization_cache():
    """Clear the visualization cache periodically."""
    global visualization_cache
    # Only keep the most recent 10 visualizations
    if len(visualization_cache) > 10:
        # Sort by timestamp and keep the most recent 10
        sorted_keys = sorted(visualization_cache.keys(), key=lambda k: visualization_cache[k]['timestamp'])
        for key in sorted_keys[:-10]:
            del visualization_cache[key]

def generate_timestamp():
    """Generate a timestamp for the current time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_complaint_intensity(probability):
    """Convert probability to intensity label."""
    if probability < 0.25:
        return "Low"
    elif probability < 0.5:
        return "Medium"
    elif probability < 0.75:
        return "High"
    else:
        return "Very High"

def create_session_id():
    """Create a unique session ID."""
    return f"session_{int(time.time())}"

def plot_to_base64(fig):
    """Convert a matplotlib figure to a base64 encoded string."""
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def process_data_chunk(session_id, texts, model):
    """Process a chunk of data and update session data."""
    # Check if session exists
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = sessions[session_id]
    
    # Create timestamps if not provided
    timestamps = [generate_timestamp() for _ in range(len(texts))]
    
    # Make predictions
    results = []
    for text, timestamp in zip(texts, timestamps):
        is_complaint, probability, _ = model.predict(text)
        intensity = get_complaint_intensity(probability)
        
        result = {
            "text": text,
            "is_complaint": bool(is_complaint),
            "complaint_probability": float(probability),
            "timestamp": timestamp,
            "complaint_intensity": intensity
        }
        results.append(result)
        
        # Add to session data
        session["data"].append(result)
    
    # Update session statistics
    session["message_count"] += len(texts)
    session["complaint_count"] += sum(1 for r in results if r["is_complaint"])
    
    # Calculate complaint percentage
    if session["message_count"] > 0:
        session["complaint_percentage"] = session["complaint_count"] / session["message_count"]
    
    return results

def generate_visualizations(session_id):
    """Generate visualizations for the session data."""
    # Check if session exists
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = sessions[session_id]
    session_data = session["data"]
    
    # Check if we have enough data
    if len(session_data) < 1:
        return {}
    
    # Convert session data to DataFrame
    df = pd.DataFrame(session_data)
    
    # Generate cache key based on session data
    cache_key = f"{session_id}_{len(session_data)}"
    
    # Check if visualizations are already cached
    if cache_key in visualization_cache:
        return visualization_cache[cache_key]["visualizations"]
    
    # Generate visualizations
    visualizations = {}
    
    # Timeline chart
    fig = plot_complaint_timeline(df, timestamp_column="timestamp")
    visualizations["timeline"] = plot_to_base64(fig)
    
    # Gauge chart
    complaint_percentage = session["complaint_percentage"]
    fig = plot_complaint_percentage_gauge(complaint_percentage)
    visualizations["gauge"] = plot_to_base64(fig)
    
    # Heatmap
    fig = plot_complaint_heatmap(df)
    visualizations["heatmap"] = plot_to_base64(fig)
    
    # Distribution
    fig = plot_complaint_distribution(df)
    visualizations["distribution"] = plot_to_base64(fig)
    
    # Cache the visualizations
    visualization_cache[cache_key] = {
        "visualizations": visualizations,
        "timestamp": time.time()
    }
    
    # Clean up cache periodically
    clear_visualization_cache()
    
    return visualizations

# API endpoints
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Return the index page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=TextResponse)
async def predict_complaint(input_data: TextInput, model: ComplaintDetector = Depends(get_global_model)):
    """
    Predict if text contains a complaint.
    Returns classification, probability, and intensity.
    """
    try:
        # Process session if provided
        if input_data.session_id:
            # Check if session exists
            if input_data.session_id not in sessions:
                raise HTTPException(status_code=404, detail=f"Session {input_data.session_id} not found")
            
            # Process as part of session
            results = process_data_chunk(input_data.session_id, [input_data.text], model)
            result = results[0]
        else:
            # Process as standalone
            is_complaint, probability, _ = model.predict(input_data.text)
            result = {
                "is_complaint": bool(is_complaint),
                "complaint_probability": float(probability),
                "text": input_data.text,
                "timestamp": generate_timestamp(),
                "complaint_intensity": get_complaint_intensity(probability)
            }
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(
    input_data: BatchTextInput, 
    generate_viz: bool = True,
    model: ComplaintDetector = Depends(get_global_model)
):
    """
    Predict complaints for a batch of texts.
    Returns classifications, probabilities, and visualizations if requested.
    """
    try:
        # Check if session exists and process as part of session
        if input_data.session_id:
            if input_data.session_id not in sessions:
                raise HTTPException(status_code=404, detail=f"Session {input_data.session_id} not found")
            
            results = process_data_chunk(input_data.session_id, input_data.texts, model)
            
            # Get complaint percentage from session
            complaint_percentage = sessions[input_data.session_id]["complaint_percentage"]
            
            # Generate visualizations if requested
            visualizations = {}
            if generate_viz:
                visualizations = generate_visualizations(input_data.session_id)
            
            return {
                "results": results,
                "complaint_percentage": complaint_percentage,
                "visualizations": visualizations
            }
        
        # Process as standalone batch
        results = []
        timestamps = [generate_timestamp() for _ in range(len(input_data.texts))]
        
        # Make predictions
        for text, timestamp in zip(input_data.texts, timestamps):
            is_complaint, probability, _ = model.predict(text)
            intensity = get_complaint_intensity(probability)
            
            results.append({
                "text": text,
                "is_complaint": bool(is_complaint),
                "complaint_probability": float(probability),
                "timestamp": timestamp,
                "complaint_intensity": intensity
            })
        
        # Calculate complaint percentage
        complaint_percentage = sum(1 for r in results if r["is_complaint"]) / len(results)
        
        # Generate visualizations if requested
        visualizations = {}
        if generate_viz:
            # Convert results to DataFrame
            df = pd.DataFrame(results)
            
            # Timeline chart
            fig = plot_complaint_timeline(df, timestamp_column="timestamp")
            visualizations["timeline"] = plot_to_base64(fig)
            
            # Gauge chart
            fig = plot_complaint_percentage_gauge(complaint_percentage)
            visualizations["gauge"] = plot_to_base64(fig)
            
            # Heatmap
            fig = plot_complaint_heatmap(df)
            visualizations["heatmap"] = plot_to_base64(fig)
            
            # Distribution
            fig = plot_complaint_distribution(df)
            visualizations["distribution"] = plot_to_base64(fig)
        
        return {
            "results": results,
            "complaint_percentage": complaint_percentage,
            "visualizations": visualizations
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

@app.post("/sessions/create", response_model=SessionInfo)
async def create_session(input_data: LiveSessionInput, model: ComplaintDetector = Depends(get_global_model)):
    """
    Create a new session for live complaint detection.
    Returns a session ID for tracking the conversation.
    """
    try:
        # Create session ID
        session_id = create_session_id()
        
        # Create session
        sessions[session_id] = {
            "name": input_data.session_name,
            "created_at": generate_timestamp(),
            "message_count": 0,
            "complaint_count": 0,
            "complaint_percentage": 0.0,
            "data": [],
            "threshold": input_data.threshold,
            "chunk_size": input_data.chunk_size,
            "chunks": []  # Store chunks of messages
        }
        
        # Update model threshold for this session
        model.threshold = input_data.threshold
        
        return {
            "session_id": session_id,
            "name": input_data.session_name,
            "created_at": sessions[session_id]["created_at"],
            "message_count": 0,
            "complaint_count": 0,
            "complaint_percentage": 0.0
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@app.get("/sessions", response_model=List[SessionInfo])
async def list_sessions():
    """
    List all active sessions.
    """
    try:
        session_info = []
        for session_id, session in sessions.items():
            session_info.append({
                "session_id": session_id,
                "name": session["name"],
                "created_at": session["created_at"],
                "message_count": session["message_count"],
                "complaint_count": session["complaint_count"],
                "complaint_percentage": session["complaint_percentage"]
            })
        return session_info
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")

@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """
    Get information about a specific session.
    """
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        session = sessions[session_id]
        return {
            "session_id": session_id,
            "name": session["name"],
            "created_at": session["created_at"],
            "message_count": session["message_count"],
            "complaint_count": session["complaint_count"],
            "complaint_percentage": session["complaint_percentage"]
        }
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error getting session: {str(e)}")

@app.post("/sessions/{session_id}/add_message")
async def add_message(
    session_id: str, 
    input_data: TextInput, 
    background_tasks: BackgroundTasks,
    model: ComplaintDetector = Depends(get_global_model)
):
    """
    Add a message to a session and process it when the chunk is complete.
    """
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        session = sessions[session_id]
        chunk_size = session["chunk_size"]
        
        # Add message to chunks
        session["chunks"].append(input_data.text)
        
        # Check if we have enough messages to process
        if len(session["chunks"]) >= chunk_size:
            # Process the chunk
            texts = session["chunks"]
            results = process_data_chunk(session_id, texts, model)
            
            # Clear the chunk
            session["chunks"] = []
            
            # Generate visualizations in the background
            background_tasks.add_task(generate_visualizations, session_id)
            
            return {
                "processed": True,
                "chunk_size": chunk_size,
                "results": results
            }
        
        # Not enough messages yet
        return {
            "processed": False,
            "chunk_size": chunk_size,
            "messages_in_chunk": len(session["chunks"]),
            "messages_needed": chunk_size - len(session["chunks"])
        }
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error adding message: {str(e)}")

@app.get("/sessions/{session_id}/visualizations")
async def get_session_visualizations(session_id: str):
    """
    Get visualizations for a session.
    """
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        # Generate visualizations
        visualizations = generate_visualizations(session_id)
        
        return {"visualizations": visualizations}
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error getting visualizations: {str(e)}")

@app.get("/sessions/{session_id}/data")
async def get_session_data(session_id: str):
    """
    Get all data for a session.
    """
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        return {
            "session_info": {
                "session_id": session_id,
                "name": sessions[session_id]["name"],
                "created_at": sessions[session_id]["created_at"],
                "message_count": sessions[session_id]["message_count"],
                "complaint_count": sessions[session_id]["complaint_count"],
                "complaint_percentage": sessions[session_id]["complaint_percentage"]
            },
            "data": sessions[session_id]["data"]
        }
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error getting session data: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session.
    """
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        # Delete session
        del sessions[session_id]
        
        # Clean up visualization cache for this session
        keys_to_delete = [k for k in visualization_cache.keys() if k.startswith(f"{session_id}_")]
        for key in keys_to_delete:
            del visualization_cache[key]
        
        return {"message": f"Session {session_id} deleted"}
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

# WebSocket for real-time processing
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    try:
        # Check if session exists
        if session_id not in sessions:
            await websocket.send_text(json.dumps({"error": f"Session {session_id} not found"}))
            await websocket.close()
            return
        
        # Get session
        session = sessions[session_id]
        chunk_size = session["chunk_size"]
        
        # Get model
        model = get_global_model()
        
        # Create buffer for messages
        buffer = []
        
        # Send initial message
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "session_id": session_id,
            "chunk_size": chunk_size
        }))
        
        # Process messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Add message to buffer
            buffer.append(message["text"])
            
            # Check if we have enough messages to process
            if len(buffer) >= chunk_size:
                # Process the chunk
                results = process_data_chunk(session_id, buffer, model)
                
                # Clear buffer
                buffer = []
                
                # Generate visualizations
                visualizations = generate_visualizations(session_id)
                
                # Send results
                await websocket.send_text(json.dumps({
                    "type": "chunk_processed",
                    "results": results,
                    "visualizations": visualizations,
                    "session_info": {
                        "message_count": session["message_count"],
                        "complaint_count": session["complaint_count"],
                        "complaint_percentage": session["complaint_percentage"]
                    }
                }))
            else:
                # Send acknowledgment
                await websocket.send_text(json.dumps({
                    "type": "message_received",
                    "messages_in_buffer": len(buffer),
                    "messages_needed": chunk_size - len(buffer)
                }))
    
    except Exception as e:
        await websocket.send_text(json.dumps({"error": str(e)}))
    finally:
        await websocket.close()

# Main function to run the API
if __name__ == "__main__":
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True) 