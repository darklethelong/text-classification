"""
Model Service Module
-------------------
This module provides the service for loading and using the complaint detection model.
"""

import json
import logging
import os
from typing import Tuple

import nltk
import torch
from nltk.tokenize import word_tokenize
from transformers import RobertaTokenizer

logger = logging.getLogger(__name__)


class ModelService:
    """Service for loading and using the complaint detection model."""
    
    def __init__(self, model_path: str):
        """Initialize the model service.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.vocab = None
        self.tokenizer = None
        self.is_roberta = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
        
    def load_model(self):
        """Load the model and vocabulary/tokenizer."""
        try:
            # Ensure NLTK data is downloaded
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            # Import project-specific modules
            sys_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            import sys
            sys.path.append(sys_path)
            
            from src.models.lstm_cnn_model import LSTMCNN
            from src.data.preprocessor import VocabBuilder
            
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
            
            # Check if we're using a RoBERTa tokenizer
            vocab_path = self.model_path.replace('.pt', '_vocab.json')
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
            
            # Check if this is a RoBERTa model
            if 'tokenizer_name' in vocab_data and vocab_data['tokenizer_name'] == 'roberta-base':
                self.is_roberta = True
                self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
                logger.info("Using RoBERTa tokenizer")
            else:
                # Use the custom vocabulary
                self.vocab = VocabBuilder()
                self.vocab.word2idx = vocab_data['word2idx']
                self.vocab.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
                logger.info(f"Using custom vocabulary with size: {len(self.vocab.word2idx)}")
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Using device: {self.device}")
            logger.info(f"Model type: {'RoBERTa' if self.is_roberta else 'Custom'}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
            
    def predict(self, text: str) -> Tuple[int, float]:
        """Make a prediction on a text input.
        
        Args:
            text: The conversation text to analyze
            
        Returns:
            Tuple containing:
                - prediction (0 for non-complaint, 1 for complaint)
                - confidence (probability of complaint class)
        
        Raises:
            RuntimeError: If model or vocabulary not loaded
            Exception: For other prediction errors
        """
        if not self.model or (not self.vocab and not self.tokenizer):
            raise RuntimeError("Model or tokenizer not loaded")
            
        try:
            # Process text based on tokenizer type
            if self.is_roberta:
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
            else:
                # Use custom vocabulary
                tokens = word_tokenize(text.lower())
                indices = [self.vocab.word2idx.get(token, 1) for token in tokens]  # 1 is <UNK>
                
                # Ensure we have at least some tokens
                if not indices:
                    return 0, 0.0
                
                # Pad sequence
                max_length = 1024  # Use a standard max length
                if len(indices) < max_length:
                    indices += [0] * (max_length - len(indices))  # 0 is <PAD>
                else:
                    indices = indices[:max_length]
                
                # Convert to tensor and move to device
                indices_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                logits = self.model(indices_tensor)
            
            # Get probabilities and prediction
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()
            prob = probs[0, 1].item()  # Probability for the positive class
            
            return pred, prob
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise 