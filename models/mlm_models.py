"""
MLM-based models for complaint detection.
Implements both full conversation and caller-only versions.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

import sys
sys.path.append('.')
import config

class MLMComplaintDetector(nn.Module):
    """
    Complaint detection model based on pretrained language model.
    Uses an MLM-based (Masked Language Model) pretrained transformer.
    """
    
    def __init__(self, model_name=config.BASE_MODEL_NAME, num_labels=2, dropout=config.MLM_DROPOUT):
        """
        Initialize model.
        
        Args:
            model_name (str): Name of pretrained model to use
            num_labels (int): Number of output labels (binary classification)
            dropout (float): Dropout rate
        """
        super(MLMComplaintDetector, self).__init__()
        
        # Load pretrained model config and adjust for classification
        model_config = AutoConfig.from_pretrained(model_name)
        model_config.num_labels = num_labels
        
        # Load base model
        self.transformer = AutoModel.from_pretrained(model_name, config=model_config)
        
        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(model_config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            labels (torch.Tensor, optional): Labels for loss calculation
            
        Returns:
            tuple: (loss, logits) or logits if no labels provided
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get CLS token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Get logits
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.classifier.out_features), labels.view(-1))
            return loss, logits
        
        return logits


def get_mlm_model(model_type="mlm_full"):
    """
    Factory function to get MLM-based models.
    
    Args:
        model_type (str): Type of model to create (mlm_full or mlm_caller_only)
        
    Returns:
        MLMComplaintDetector: Initialized model
    """
    # Both full and caller-only use the same model architecture
    # The difference is in the input data preparation (done elsewhere)
    model = MLMComplaintDetector(
        model_name=config.BASE_MODEL_NAME,
        num_labels=2,
        dropout=config.MLM_DROPOUT
    )
    
    return model 