"""
Model factory for complaint detection models.
Provides a unified interface for creating different model types.
"""

import torch
import os

import sys
sys.path.append('.')
import config
from models.mlm_models import get_mlm_model
from models.lstm_cnn_models import get_lstm_cnn_model
from models.hybrid_models import get_hybrid_model


def get_model(model_type, vocab_size=30000):
    """
    Factory function to get models of different types.
    
    Args:
        model_type (str): Type of model to create
        vocab_size (int): Size of vocabulary (only used for LSTM-CNN models)
        
    Returns:
        torch.nn.Module: Initialized model
    """
    if model_type.startswith("mlm_"):
        return get_mlm_model(model_type)
    elif model_type.startswith("lstm_cnn_"):
        return get_lstm_cnn_model(model_type, vocab_size)
    elif model_type.startswith("hybrid_"):
        return get_hybrid_model(model_type)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_model(model, model_type, save_dir=config.OUTPUT_DIR, epoch=None):
    """
    Save model checkpoint.
    
    Args:
        model (torch.nn.Module): Model to save
        model_type (str): Type of model
        save_dir (str): Directory to save model in
        epoch (int, optional): Current epoch
        
    Returns:
        str: Path to saved model
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine filename
    filename = f"{model_type}"
    if epoch is not None:
        filename += f"_epoch{epoch}"
    filename += ".pt"
    
    save_path = os.path.join(save_dir, filename)
    
    # Save model
    torch.save(model.state_dict(), save_path)
    
    return save_path


def load_model(model_type, load_path, vocab_size=30000):
    """
    Load model from checkpoint.
    
    Args:
        model_type (str): Type of model
        load_path (str): Path to model checkpoint
        vocab_size (int): Size of vocabulary (only used for LSTM-CNN models)
        
    Returns:
        torch.nn.Module: Loaded model
    """
    # Initialize model
    model = get_model(model_type, vocab_size)
    
    # Load weights
    model.load_state_dict(torch.load(load_path))
    
    return model


def get_model_info(model_type):
    """
    Get information about a model type.
    
    Args:
        model_type (str): Type of model
        
    Returns:
        dict: Model information
    """
    model_info = {
        "model_type": model_type,
        "base_model": config.BASE_MODEL_NAME if model_type.startswith(("mlm_", "hybrid_")) else "custom",
        "data_type": "full" if "full" in model_type else "caller_only",
        "architecture": model_type.split("_")[0] if "_" in model_type else model_type
    }
    
    return model_info 