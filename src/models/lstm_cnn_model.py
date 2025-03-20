import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256, num_layers=2, 
                 cnn_out_channels=100, kernel_sizes=[3, 5, 7], num_classes=2, dropout=0.5, 
                 padding_idx=0, pretrained_embeddings=None):
        super(LSTMCNN, self).__init__()
        
        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, 
                padding_idx=padding_idx,
                freeze=False
            )
            embedding_dim = pretrained_embeddings.size(1)
        else:
            self.embedding = nn.Embedding(
                vocab_size, 
                embedding_dim, 
                padding_idx=padding_idx
            )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            bidirectional=True, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # CNN layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim * 2, cnn_out_channels, kernel_size)
            for kernel_size in kernel_sizes
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Linear layers for classification
        self.fc1 = nn.Linear(len(kernel_sizes) * cnn_out_channels, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Store configurations
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, sequence_length, hidden_dim*2)
        
        # Reshape for CNN
        lstm_out = lstm_out.permute(0, 2, 1)  # (batch_size, hidden_dim*2, sequence_length)
        
        # CNN with different kernel sizes and max pooling
        conv_results = []
        for conv in self.convs:
            conv_out = F.relu(conv(lstm_out))  # (batch_size, cnn_out_channels, seq_len - kernel_size + 1)
            # Global max pooling
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch_size, cnn_out_channels)
            conv_results.append(pooled)
        
        # Concatenate results from different kernel sizes
        concatenated = torch.cat(conv_results, dim=1)  # (batch_size, num_kernels * cnn_out_channels)
        
        # Apply dropout for regularization
        concatenated = self.dropout(concatenated)
        
        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(concatenated))
        x = self.dropout(x)
        
        # Output layer
        logits = self.fc2(x)
        
        return logits 