"""
Simple Complaint Detection API Server
-------------------------------------
This is a simplified API server that loads the LSTM-CNN RoBERTa model
without complex dependencies.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Tuple

import torch
import nltk
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from transformers import RobertaTokenizer
from pydantic import BaseModel, Field

# Add src directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.append(project_root)
sys.path.append(src_path)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "output/models_20250320_154943/lstm_cnn_roberta_full.pt"
MODEL_VOCAB_PATH = "output/models_20250320_154943/lstm_cnn_roberta_full_vocab.json"
API_SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"

# Define models for API
class ConversationInput(BaseModel):
    """Input model for conversation analysis."""
    text: str = Field(..., description="The conversation text to analyze")

class ConversationOutput(BaseModel):
    """Output model for conversation analysis."""
    is_complaint: bool = Field(..., description="Whether the conversation is a complaint")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    
class ModelManager:
    """Manager for the complaint detection model."""
    def __init__(self, model_path: str, vocab_path: str):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
        
    def load_model(self):
        """Load the model and tokenizer."""
        try:
            # Ensure NLTK data is downloaded
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            # Import the model class
            from src.models.lstm_cnn_model import LSTMCNN
            
            # Load model metadata
            metadata_path = self.model_path.replace('.pt', '_metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create model with same parameters
            self.model = LSTMCNN(
                vocab_size=metadata['vocab_size'],
                embedding_dim=metadata['embedding_dim'],
                hidden_dim=metadata['hidden_dim'],
                num_layers=metadata['num_layers'],
                cnn_out_channels=metadata['cnn_out_channels'],
                kernel_sizes=metadata['kernel_sizes'],
                dropout=metadata['dropout']
            )
            
            # Load model weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load tokenizer
            with open(self.vocab_path, 'r') as f:
                vocab_data = json.load(f)
            
            # Check if this is a RoBERTa model
            if 'tokenizer_name' in vocab_data and vocab_data['tokenizer_name'] == 'roberta-base':
                self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
                logger.info("Using RoBERTa tokenizer")
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def predict(self, text: str) -> Tuple[bool, float]:
        """Make a prediction on the input text."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model or tokenizer not loaded")
            
        try:
            # Use RoBERTa tokenizer
            max_length = 512
            encoded = self.tokenizer.encode_plus(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            indices_tensor = encoded['input_ids'].to(self.device)
            
            # Get prediction
            with torch.no_grad():
                logits = self.model(indices_tensor)
            
            # Get probabilities and prediction
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()
            prob = probs[0, 1].item()  # Probability for the positive class
            
            return bool(pred), prob
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

# Create OAuth2 scheme for token authentication (simple version)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize the app
app = FastAPI(
    title="Complaint Detection API (Simplified)",
    description="API for detecting complaints in customer service conversations using LSTM-CNN with RoBERTa",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model manager
model_manager = ModelManager(MODEL_PATH, MODEL_VOCAB_PATH)

# Simple token validation for authentication
async def verify_token(token: str = Depends(oauth2_scheme)):
    """Simple token verification for demo purposes."""
    if token != "demo_token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

@app.post("/analyze", response_model=ConversationOutput)
async def analyze_conversation(
    conversation: ConversationInput,
    token: str = Depends(verify_token)
):
    """Analyze a conversation to detect if it's a complaint."""
    try:
        is_complaint, confidence = model_manager.predict(conversation.text)
        return {
            "is_complaint": is_complaint,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error analyzing conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing conversation: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint for API health check."""
    return {
        "status": "online",
        "model": "LSTM-CNN with RoBERTa tokenizer",
        "auth_required": "Bearer Token (use 'demo_token')"
    }

# Create a simple token endpoint
@app.post("/token")
async def get_token():
    """Get a demo token for API access."""
    return {"access_token": "demo_token", "token_type": "bearer"}

if __name__ == "__main__":
    print(f"Starting Complaint Detection API server with LSTM-CNN RoBERTa model...")
    print(f"Model path: {MODEL_PATH}")
    print(f"API Documentation: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000) 