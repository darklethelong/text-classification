"""
Hybrid models for complaint detection.
Combines pretrained transformer models with LSTM and CNN layers.
Implements both full conversation and caller-only versions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

import sys
sys.path.append('.')
import config


class HybridComplaintDetector(nn.Module):
    """
    Hybrid complaint detection model.
    Combines a pretrained transformer model with LSTM and CNN layers.
    """
    
    def __init__(self, model_name=config.BASE_MODEL_NAME, lstm_units=config.HYBRID_LSTM_UNITS,
                 cnn_filters=config.HYBRID_CNN_FILTERS, kernel_sizes=config.CNN_KERNEL_SIZES,
                 dropout=config.HYBRID_DROPOUT, num_labels=2):
        """
        Initialize hybrid model.
        
        Args:
            model_name (str): Name of pretrained model
            lstm_units (int): Number of LSTM units
            cnn_filters (int): Number of CNN filters
            kernel_sizes (list): List of kernel sizes for CNN
            dropout (float): Dropout rate
            num_labels (int): Number of output labels
        """
        super(HybridComplaintDetector, self).__init__()
        
        # Load pretrained model config
        model_config = AutoConfig.from_pretrained(model_name)
        self.hidden_size = model_config.hidden_size
        
        # Load base transformer model
        self.transformer = AutoModel.from_pretrained(model_name, config=model_config)
        
        # Add LSTM layer
        self.lstm = nn.LSTM(
            self.hidden_size,
            lstm_units,
            bidirectional=True,
            batch_first=True
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = lstm_units * 2  # bidirectional
        
        # Add CNN layers
        self.convs = nn.ModuleList([
            nn.Conv1d(lstm_output_dim, cnn_filters, kernel_size)
            for kernel_size in kernel_sizes
        ])
        
        # Add dropout
        self.dropout = nn.Dropout(dropout)
        
        # Add classifier
        self.classifier = nn.Linear(len(kernel_sizes) * cnn_filters, num_labels)
        
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
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get sequence of hidden states
        hidden_states = transformer_outputs.last_hidden_state
        
        # Apply LSTM
        lstm_output, _ = self.lstm(hidden_states)
        
        # Apply CNN layers
        # Transpose for conv1d which expects (batch_size, channels, seq_len)
        conv_input = lstm_output.permute(0, 2, 1)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            # Apply convolution
            conv_out = F.relu(conv(conv_input))
            # Apply max pooling over time
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # Concatenate outputs from different kernels
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # Apply dropout
        features = self.dropout(concatenated)
        
        # Get logits
        logits = self.classifier(features)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.classifier.out_features), labels.view(-1))
            return loss, logits
        
        return logits


def get_hybrid_model(model_type="hybrid_full"):
    """
    Factory function to get hybrid models.
    
    Args:
        model_type (str): Type of model to create (hybrid_full or hybrid_caller_only)
        
    Returns:
        HybridComplaintDetector: Initialized model
    """
    # Both full and caller-only use the same model architecture
    # The difference is in the input data preparation (done elsewhere)
    model = HybridComplaintDetector(
        model_name=config.BASE_MODEL_NAME,
        lstm_units=config.HYBRID_LSTM_UNITS,
        cnn_filters=config.HYBRID_CNN_FILTERS,
        kernel_sizes=config.CNN_KERNEL_SIZES,
        dropout=config.HYBRID_DROPOUT,
        num_labels=2
    )
    
    return model 