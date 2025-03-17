"""
LSTM-CNN models for complaint detection.
Implements both full conversation and caller-only versions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('.')
import config

class CNNBlock(nn.Module):
    """CNN block for text classification."""
    
    def __init__(self, in_channels, out_channels, kernel_sizes, dropout=0.2):
        """
        Initialize CNN block.
        
        Args:
            in_channels (int): Input channels
            out_channels (int): Output channels per kernel
            kernel_sizes (list): List of kernel sizes
            dropout (float): Dropout rate
        """
        super(CNNBlock, self).__init__()
        
        # Create a CNN layer for each kernel size
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size)
            for kernel_size in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, in_channels)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Transpose for conv1d which expects (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            # Apply convolution
            conv_out = F.relu(conv(x))
            # Apply max pooling over time
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # Concatenate outputs from different kernels
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # Apply dropout
        output = self.dropout(concatenated)
        
        return output


class LSTMCNNComplaintDetector(nn.Module):
    """
    Complaint detection model using LSTM and CNN architectures.
    """
    
    def __init__(self, vocab_size, embedding_dim=config.LSTM_EMBEDDING_DIM, 
                 hidden_dim=config.LSTM_HIDDEN_DIM, num_layers=config.LSTM_LAYERS,
                 bidirectional=config.LSTM_BIDIRECTIONAL, dropout=config.LSTM_DROPOUT,
                 cnn_filters=config.CNN_FILTERS, kernel_sizes=config.CNN_KERNEL_SIZES,
                 num_labels=2):
        """
        Initialize LSTM-CNN model.
        
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Embedding dimension
            hidden_dim (int): LSTM hidden dimension
            num_layers (int): Number of LSTM layers
            bidirectional (bool): Whether to use bidirectional LSTM
            dropout (float): Dropout rate
            cnn_filters (int): Number of CNN filters per kernel
            kernel_sizes (list): List of kernel sizes for CNN
            num_labels (int): Number of output labels
        """
        super(LSTMCNNComplaintDetector, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # CNN block
        self.cnn_block = CNNBlock(
            in_channels=lstm_output_dim,
            out_channels=cnn_filters,
            kernel_sizes=kernel_sizes,
            dropout=dropout
        )
        
        # Output layer
        self.classifier = nn.Linear(len(kernel_sizes) * cnn_filters, num_labels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor, optional): Attention mask
            labels (torch.Tensor, optional): Labels for loss calculation
            
        Returns:
            tuple: (loss, logits) or logits if no labels provided
        """
        # Get embeddings
        embedded = self.embedding(input_ids)
        
        # Apply LSTM
        lstm_output, _ = self.lstm(embedded)
        
        # Apply CNN block
        cnn_output = self.cnn_block(lstm_output)
        
        # Get logits
        logits = self.classifier(cnn_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.classifier.out_features), labels.view(-1))
            return loss, logits
        
        return logits


def get_lstm_cnn_model(model_type="lstm_cnn_full", vocab_size=30000):
    """
    Factory function to get LSTM-CNN models.
    
    Args:
        model_type (str): Type of model to create (lstm_cnn_full or lstm_cnn_caller_only)
        vocab_size (int): Size of vocabulary
        
    Returns:
        LSTMCNNComplaintDetector: Initialized model
    """
    # Both full and caller-only use the same model architecture
    # The difference is in the input data preparation (done elsewhere)
    model = LSTMCNNComplaintDetector(
        vocab_size=vocab_size,
        embedding_dim=config.LSTM_EMBEDDING_DIM,
        hidden_dim=config.LSTM_HIDDEN_DIM,
        num_layers=config.LSTM_LAYERS,
        bidirectional=config.LSTM_BIDIRECTIONAL,
        dropout=config.LSTM_DROPOUT,
        cnn_filters=config.CNN_FILTERS,
        kernel_sizes=config.CNN_KERNEL_SIZES,
        num_labels=2
    )
    
    return model 