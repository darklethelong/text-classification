import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import json
import datetime

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define constants
MAX_LEN = 1024
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VOCAB_SIZE = 30000 
EMBEDDING_DIM = 128

# Custom dataset class for conversation data
class ConversationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Convert string labels to integers
        self.labels = []
        for label in labels:
            if isinstance(label, str):
                # Convert string labels to binary
                if label.lower() in ['complaint', 'complaints', '1', 'true', 'yes']:
                    self.labels.append(1)
                else:  # 'non-complaint', 'non_complaint', '0', 'false', 'no', etc.
                    self.labels.append(0)
            else:
                # Assume it's already a numeric type
                self.labels.append(int(label))
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# LSTM-CNN hybrid model for complaint classification
class LSTM_CNN_Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_size=128, cnn_out_channels=64, num_classes=2):
        super(LSTM_CNN_Classifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # CNN layers with different kernel sizes for capturing different n-gram features
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=lstm_hidden_size*2, out_channels=cnn_out_channels, kernel_size=k)
            for k in [3, 4, 5]
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(cnn_out_channels * 3, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(64)
    
    def forward(self, input_ids, attention_mask):
        # Embedding
        embedded = self.embedding(input_ids)  # Shape: [batch_size, seq_len, embedding_dim]
        
        # Apply attention mask
        embedded = embedded * attention_mask.unsqueeze(2)
        
        # Pass through LSTM
        lstm_output, _ = self.lstm(embedded)  # Shape: [batch_size, seq_len, lstm_hidden_size*2]
        
        # Apply attention mask again
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

# Main training and evaluation function
def train_and_evaluate(train_df, test_df=None, test_size=0.1):
    # If test_df is not provided, split train_df into train and test
    if test_df is None:
        # Convert labels if needed for stratification
        if 'label' in train_df.columns and train_df['label'].dtype == 'object':
            # Create temporary numeric labels for stratification
            temp_labels = train_df['label'].apply(lambda x: 1 if x.lower() in ['complaint', 'complaints', '1', 'true', 'yes'] else 0)
            train_df, test_df = train_test_split(train_df, test_size=test_size, random_state=42, stratify=temp_labels)
        else:
            train_df, test_df = train_test_split(train_df, test_size=test_size, random_state=42, stratify=train_df['label'])
    
    # Prepare data
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    
    # Create datasets
    train_dataset = ConversationDataset(
        texts=train_df['text'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    test_dataset = ConversationDataset(
        texts=test_df['text'].values,
        labels=test_df['label'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = LSTM_CNN_Classifier(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM
    ).to(DEVICE)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop
    best_f1 = 0  # Changed from accuracy to F1 score for best model selection
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate on validation set
        avg_loss = total_loss / len(train_loader)
        metrics = evaluate(model, test_loader, criterion)
        
        # Update learning rate
        scheduler.step(metrics['loss'])
        
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'Training Loss: {avg_loss:.4f}')
        print(f'Validation Loss: {metrics["loss"]:.4f}')
        print(f'Validation Metrics:')
        print(f'  Accuracy: {metrics["accuracy"]:.4f}')
        print(f'  Precision: {metrics["precision"]:.4f}')
        print(f'  Recall: {metrics["recall"]:.4f}')
        print(f'  F1: {metrics["f1"]:.4f}')
        
        # Save best model based on F1 score
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), 'best_model.pt')
            print(f'New best model saved with F1: {best_f1:.4f}')
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load('best_model.pt'))
    final_metrics = evaluate(model, test_loader, criterion, detailed=True)
    print(f'Final Evaluation Metrics:')
    print(f'  Accuracy: {final_metrics["accuracy"]:.4f}')
    print(f'  Precision: {final_metrics["precision"]:.4f}')
    print(f'  Recall: {final_metrics["recall"]:.4f}')
    print(f'  F1: {final_metrics["f1"]:.4f}')

    # Save metadata for API server
    metadata = {
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": 256,  # lstm_hidden_size
        "num_layers": 2,
        "cnn_out_channels": 128,
        "kernel_sizes": [3, 4, 5],
        "dropout": 0.3,
        "tokenizer_name": "albert-base-v2"
    }
    
    # Save complete model with metadata
    output_dir = "output/models_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "lstm_cnn_roberta_full.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': final_metrics,
        'epoch': EPOCHS
    }, model_path)
    
    # Save metadata separately
    metadata_path = os.path.join(output_dir, "lstm_cnn_roberta_full_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Save vocab info
    vocab_path = os.path.join(output_dir, "lstm_cnn_roberta_full_vocab.json")
    with open(vocab_path, 'w') as f:
        json.dump({"tokenizer_name": "albert-base-v2"}, f, indent=4)
    
    print(f"Model and metadata saved to {output_dir}")
    
    return model, tokenizer

# Evaluation function
def evaluate(model, data_loader, criterion, detailed=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    avg_loss = total_loss / len(data_loader)
    
    if detailed:
        print('\nMetrics:')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        
        print('\nClassification Report:')
        print(classification_report(all_labels, all_preds, target_names=['Non-Complaint', 'Complaint']))
        
        print('\nConfusion Matrix:')
        print(confusion_matrix(all_labels, all_preds))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': avg_loss
    }

# Function to use the trained model for prediction
def predict_complaint(text, model, tokenizer, max_len=MAX_LEN):
    model.eval()
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = F.softmax(outputs, dim=1)
        confidence, preds = torch.max(probs, dim=1)
    
    result = {
        "prediction": "Complaint" if preds.item() == 1 else "Non-Complaint",
        "confidence": confidence.item(),
        "probabilities": {
            "Non-Complaint": probs[0, 0].item(),
            "Complaint": probs[0, 1].item()
        }
    }
    
    return result

# Example usage:
if __name__ == "__main__":
    # Check if specific files exist first
    if os.path.exists("train.csv"):
        print("Loading data from train.csv...")
        df = pd.read_csv("train.csv")
        # Train and evaluate the model
        model, tokenizer = train_and_evaluate(df)
    else:
        print("No train.csv found. Using sample data...")
        # Sample data structure
        sample_data = {
            'text': [
                "Agent: Thank you for calling. How may I assist you? Caller: Hi. I purchased your product and I'm having trouble. Agent: I understand. Could you provide your order number? Caller: Yes, it's THG-29875.",
                "Agent: Thanks for that. You purchased our SmartHome Hub? Caller: Yes, and it's not connecting to anything. I've followed all instructions. Agent: Let's troubleshoot. Have you downloaded our app? Caller: Yes, I've updated it twice. I'm getting quite annoyed.",
                "Agent: I understand your frustration. Is the hub connected to Wi-Fi? Caller: The light keeps blinking green. This product is clearly defective! Agent: I'm sorry to hear that. Could you tell me which version you have? Caller: Version 2.1.4. I paid $200 for this and it doesn't work. This is completely unacceptable.",
                # Add more examples as needed
            ],
            'label': ["non-complaint", "complaint", "complaint"]  # String labels that will be converted
        }
        
        # Convert to DataFrame
        df = pd.DataFrame(sample_data)
        # Train and evaluate the model
        model, tokenizer = train_and_evaluate(df)
    
    # Example prediction
    test_text = "Agent: Let's try recovery mode. Caller: Okay, it's searching now. Agent: The app should install updated firmware. Caller: This should have worked right out of the box."
    prediction = predict_complaint(test_text, model, tokenizer)
    print(f"Prediction: {prediction}")
