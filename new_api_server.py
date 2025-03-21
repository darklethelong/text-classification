"""
Advanced Complaint Detection API Server
--------------------------------------
This API server processes both single chunks and full conversations,
analyzing them with the LSTM-CNN RoBERTa model.
"""

import os
import sys
import json
import time
import logging
import datetime
import re
from typing import Dict, List, Optional, Tuple, Any

import torch
import nltk
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from transformers import RobertaTokenizer
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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
class UtteranceModel(BaseModel):
    """Model for a single utterance."""
    speaker: str = Field(..., description="The speaker (agent or caller)")
    content: str = Field(..., description="The utterance text")

class ChunkModel(BaseModel):
    """Model for a conversation chunk."""
    id: str = Field(..., description="Unique identifier for the chunk")
    utterances: List[UtteranceModel] = Field(..., description="List of utterances in the chunk")
    text: str = Field(..., description="Full text of the chunk")
    is_complaint: bool = Field(False, description="Whether the chunk is a complaint")
    confidence: float = Field(0.0, description="Confidence score for complaint classification")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat(), 
                          description="Timestamp when the chunk was analyzed")
    window_start: Optional[int] = Field(None, description="Starting position of the window (1-indexed)")
    window_end: Optional[int] = Field(None, description="Ending position of the window (1-indexed)")

class ConversationInput(BaseModel):
    """Input model for conversation analysis."""
    text: str = Field(..., description="The conversation text to analyze")
    chunk_size: int = Field(4, description="Number of utterances per chunk (default: 4)")

class ChunkInput(BaseModel):
    """Input model for single chunk analysis."""
    text: str = Field(..., description="The conversation chunk to analyze")

class ConversationOutput(BaseModel):
    """Output model for full conversation analysis."""
    chunks: List[ChunkModel] = Field(..., description="Analysis results for each conversation chunk")
    complaint_percentage: float = Field(..., description="Percentage of chunks classified as complaints")
    
class ChunkOutput(BaseModel):
    """Output model for single chunk analysis."""
    is_complaint: bool = Field(..., description="Whether the chunk is a complaint")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    timestamp: str = Field(..., description="Timestamp when the chunk was analyzed")

class ConversationProcessor:
    """Processes conversations into chunks and utterances."""
    
    @staticmethod
    def parse_conversation(text: str) -> List[UtteranceModel]:
        """Parse a conversation string into separate utterances."""
        # Split by new lines
        lines = text.split('\n')
        utterances = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to identify the speaker (Caller/Agent)
            speaker_pattern = r'^(Caller|Agent|Customer|Representative|Rep|Support|User|System):\s*(.*)'
            speaker_match = re.search(speaker_pattern, line, re.IGNORECASE)
            
            if speaker_match:
                speaker = speaker_match.group(1).lower()
                content = speaker_match.group(2).strip()
                
                # Normalize speaker names
                if speaker in ['customer', 'user']:
                    speaker = 'caller'
                elif speaker in ['representative', 'rep', 'support', 'system']:
                    speaker = 'agent'
                
                utterances.append(UtteranceModel(
                    speaker=speaker,
                    content=content
                ))
            else:
                # If no speaker is identified, append to the previous utterance
                if utterances:
                    utterances[-1].content += ' ' + line
                else:
                    # If this is the first line and no speaker, assume it's the caller
                    utterances.append(UtteranceModel(
                        speaker='caller',
                        content=line
                    ))
        
        logger.info(f"Parsed {len(utterances)} utterances from conversation text")
        for i, u in enumerate(utterances):
            logger.info(f"Utterance {i+1}: {u.speaker}: {u.content[:30]}...")
            
        return utterances
    
    @staticmethod
    def chunk_conversation(utterances: List[UtteranceModel], chunk_size: int = 4) -> List[ChunkModel]:
        """Split a conversation into chunks using a sliding window approach.
        This ensures all possible groups of consecutive utterances are analyzed.
        """
        chunks = []
        
        logger.info(f"Chunking conversation with {len(utterances)} utterances using chunk_size={chunk_size}")
        
        # If we have fewer utterances than chunk_size, we can't make a complete chunk
        if len(utterances) < chunk_size:
            if len(utterances) > 0:
                # Create a partial chunk with whatever utterances we have
                chunk_text = '\n'.join([f"{u.speaker}: {u.content}" for u in utterances])
                chunk = ChunkModel(
                    id="chunk-0",
                    utterances=utterances,
                    text=chunk_text,
                    is_complaint=False,
                    confidence=0.0
                )
                chunks.append(chunk)
                logger.info(f"Created partial chunk with {len(utterances)} utterances (fewer than chunk_size)")
            return chunks
        
        # Use sliding window approach - each possible starting position
        expected_chunks = len(utterances) - chunk_size + 1
        logger.info(f"Expecting to create {expected_chunks} chunks with sliding window")
        
        for i in range(len(utterances) - chunk_size + 1):
            chunk_utterances = utterances[i:i+chunk_size]
            
            # Create chunk text
            chunk_text = '\n'.join([f"{u.speaker}: {u.content}" for u in chunk_utterances])
            
            # Create chunk model with window information
            chunk = ChunkModel(
                id=f"chunk-{i+1}",  # 1-indexed for display
                utterances=chunk_utterances,
                text=chunk_text,
                is_complaint=False,  # Will be updated after prediction
                confidence=0.0,      # Will be updated after prediction
                window_start=i+1,    # 1-indexed position
                window_end=i+chunk_size  # Last position in window
            )
            
            chunks.append(chunk)
            logger.info(f"Created chunk {i+1}/{expected_chunks} with window positions {i+1}-{i+chunk_size}")
        
        logger.info(f"Created total of {len(chunks)} chunks")
        return chunks
    
    @staticmethod
    def ensure_chunk_size(text: str, chunk_size: int = 4) -> Tuple[bool, str]:
        """Ensure that a chunk has exactly chunk_size utterances."""
        utterances = ConversationProcessor.parse_conversation(text)
        
        if len(utterances) == chunk_size:
            return True, text
            
        if len(utterances) < chunk_size:
            missing = chunk_size - len(utterances)
            logger.warning(f"Chunk has only {len(utterances)} utterances, {missing} more needed")
            return False, f"Chunk must have exactly {chunk_size} utterances (found {len(utterances)})"
            
        if len(utterances) > chunk_size:
            logger.warning(f"Chunk has {len(utterances)} utterances, only using first {chunk_size}")
            
            # Truncate to chunk_size utterances
            truncated = utterances[:chunk_size]
            truncated_text = '\n'.join([f"{u.speaker}: {u.content}" for u in truncated])
            
            return True, truncated_text
    
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

    def analyze_conversation(self, text: str, chunk_size: int = 4) -> ConversationOutput:
        """Analyze a full conversation."""
        try:
            logger.info(f"Analyzing conversation with chunk_size={chunk_size}")
            logger.info(f"Conversation length (chars): {len(text)}")
            logger.info(f"Estimated utterances: ~{text.count('Caller:') + text.count('Agent:')}")
            
            # Parse the conversation into utterances
            utterances = ConversationProcessor.parse_conversation(text)
            logger.info(f"Parsed {len(utterances)} utterances")
            
            # Split into chunks
            chunks = ConversationProcessor.chunk_conversation(utterances, chunk_size)
            logger.info(f"Created {len(chunks)} chunks for analysis")
            
            # Analyze each chunk
            for i, chunk in enumerate(chunks):
                logger.info(f"Analyzing chunk {i+1}/{len(chunks)} (ID: {chunk.id})")
                is_complaint, confidence = self.predict(chunk.text)
                chunk.is_complaint = is_complaint
                chunk.confidence = confidence
                chunk.timestamp = datetime.datetime.now().isoformat()
                logger.info(f"Chunk {chunk.id} analysis: is_complaint={is_complaint}, confidence={confidence:.4f}")
            
            # Calculate complaint percentage
            if chunks:
                complaint_chunks = [c for c in chunks if c.is_complaint]
                complaint_percentage = (len(complaint_chunks) / len(chunks)) * 100
                logger.info(f"Overall analysis: {len(complaint_chunks)}/{len(chunks)} chunks are complaints ({complaint_percentage:.2f}%)")
            else:
                complaint_percentage = 0.0
                logger.warning("No chunks were created, setting complaint percentage to 0")
            
            return ConversationOutput(
                chunks=chunks,
                complaint_percentage=complaint_percentage
            )
            
        except Exception as e:
            logger.error(f"Conversation analysis error: {str(e)}")
            logger.exception("Full traceback:")
            raise

# Create OAuth2 scheme for token authentication (simple version)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize the app
app = FastAPI(
    title="Advanced Complaint Detection API",
    description="API for detecting complaints in customer service conversations with support for chunk-based and full conversation analysis",
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

# Mount static files for UI
app.mount("/static", StaticFiles(directory="static"), name="static")

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

@app.post("/analyze/chunk", response_model=ChunkOutput)
async def analyze_chunk(
    chunk: ChunkInput,
    token: str = Depends(verify_token)
):
    """Analyze a single conversation chunk."""
    try:
        # Ensure the chunk has exactly 4 utterances
        is_valid, result = ConversationProcessor.ensure_chunk_size(chunk.text)
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result
            )
        
        # Use the validated/truncated chunk
        is_complaint, confidence = model_manager.predict(result)
        timestamp = datetime.datetime.now().isoformat()
        
        return ChunkOutput(
            is_complaint=is_complaint,
            confidence=confidence,
            timestamp=timestamp
        )
        
    except Exception as e:
        logger.error(f"Error analyzing chunk: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing chunk: {str(e)}"
        )

@app.post("/analyze/conversation", response_model=ConversationOutput)
async def analyze_conversation(
    conversation: ConversationInput,
    token: str = Depends(verify_token)
):
    """Analyze a full conversation, processing chunks of 4 utterances."""
    try:
        return model_manager.analyze_conversation(conversation.text, conversation.chunk_size)
        
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
        "description": "Advanced Complaint Detection API with support for chunk and conversation analysis",
        "auth_required": "Bearer Token (use 'demo_token')"
    }

# Create a simple token endpoint
@app.post("/token")
async def get_token():
    """Get a demo token for API access."""
    return {"access_token": "demo_token", "token_type": "bearer"}

# Add a route to serve the UI
@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse('static/index.html')

# Add catch-all route to handle React Router paths
@app.get("/{path:path}", include_in_schema=False)
async def serve_ui_paths(path: str):
    # Exclude API paths
    if path.startswith("api/") or path.startswith("docs") or path.startswith("openapi.json"):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse('static/index.html')

# Add Health Check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "api_version": "1.0.0"}

if __name__ == "__main__":
    print(f"Starting Advanced Complaint Detection API server with LSTM-CNN RoBERTa model...")
    print(f"Model path: {MODEL_PATH}")
    print(f"API Documentation: http://127.0.0.1:8000/docs")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("new_api_server:app", host=host, port=port, reload=False) 