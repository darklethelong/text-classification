import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_CNN_Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_size=256, cnn_out_channels=128, num_classes=2, dropout=0.3):
        super(LSTM_CNN_Classifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM layer - using exact dimensions from metadata
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # CNN layers with different kernel sizes for capturing different n-gram features
        # Using cnn_out_channels=128 from metadata
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=lstm_hidden_size*2, out_channels=cnn_out_channels, kernel_size=k)
            for k in [3, 4, 5]
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        # Input to fc1 is cnn_out_channels * 3 = 128 * 3 = 384
        self.fc1 = nn.Linear(cnn_out_channels * 3, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(64)
    
    def forward(self, input_ids, attention_mask=None):
        # Embedding
        embedded = self.embedding(input_ids)  # Shape: [batch_size, seq_len, embedding_dim]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            embedded = embedded * attention_mask.unsqueeze(2)
        
        # Pass through LSTM
        lstm_output, _ = self.lstm(embedded)  # Shape: [batch_size, seq_len, lstm_hidden_size*2]
        
        # Apply attention mask again if provided
        if attention_mask is not None:
            lstm_output = lstm_output * attention_mask.unsqueeze(2)
        
        # Prepare for CNN (convert to [batch_size, channels, seq_len])
        lstm_output = lstm_output.permute(0, 2, 1)
        
        # Apply CNN layers with different kernel sizes
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(lstm_output))  # Apply convolution
            conv_out = F.max_pool1d(conv_out, conv_out.shape[2])  # Global max pooling
            conv_outputs.append(conv_out.squeeze(2))
        
        # Concatenate CNN outputs
        x = torch.cat(conv_outputs, dim=1)
        
        # Pass through fully connected layers
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(self.bn(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    # Add a compatibility method for the server to work without changing too much code
    def forward_simple(self, input_ids):
        """Compatibility method that doesn't require attention_mask for simple inference."""
        attention_mask = torch.ones_like(input_ids)
        return self.forward(input_ids, attention_mask) 