"""
Complaints Router
---------------
This module contains routes for complaint detection.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from config import settings
from schemas.auth import User
from schemas.conversations import ConversationRequest, ConversationResponse
from services.auth import get_current_active_user
from services.model_service import ModelService

logger = logging.getLogger(__name__)

# Create router with prefix and tags
router = APIRouter(
    prefix=settings.api_prefix,
    tags=["complaints"]
)

# Create model service instance
model_service = ModelService(settings.model_path)


@router.post("/predict", response_model=ConversationResponse)
async def predict_complaint(
    conversation: ConversationRequest, 
    current_user: User = Depends(get_current_active_user)
):
    """
    Analyze a conversation and detect if it contains a complaint.
    
    This endpoint takes a conversation text and returns whether it contains a complaint,
    along with a confidence score.
    """
    try:
        logger.info(f"Processing prediction request from user: {current_user.username}")
        is_complaint, confidence = model_service.predict(conversation.text)
        
        result = {
            "is_complaint": bool(is_complaint),
            "confidence": float(confidence),
            "processed_at": datetime.now().isoformat()
        }
        
        logger.info(f"Prediction result: is_complaint={is_complaint}, confidence={confidence:.4f}")
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    This endpoint can be used to verify the API is running and the model is loaded.
    """
    return {
        "status": "healthy", 
        "model_loaded": model_service.model is not None,
        "vocabulary_size": len(model_service.vocab.word2idx) if model_service.vocab else 0,
        "timestamp": datetime.now().isoformat()
    } 