import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json
import os
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer

def load_tokenizer_or_vocab(vocab_path, model_type):
    """
    Load tokenizer or vocabulary based on the model type.
    
    Args:
        vocab_path: Path to vocabulary file
        model_type: Type of model ('mlm' or 'lstm-cnn')
        
    Returns:
        Tokenizer for MLM models or vocabulary for LSTM-CNN models
    """
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    if model_type.lower() == 'mlm':
        # For MLM models, load tokenizer
        return AutoTokenizer.from_pretrained(vocab_path)
    else:
        # For LSTM-CNN models, load vocabulary
        with open(vocab_path, 'r') as f:
            vocab_dict = json.load(f)
        
        # Check if it's a RoBERTa tokenizer
        if "tokenizer_name" in vocab_dict:
            # Load RoBERTa tokenizer
            return AutoTokenizer.from_pretrained(vocab_dict["tokenizer_name"])
        else:
            # Create a vocabulary object for custom vocab
            class Vocabulary:
                def __init__(self, word2idx, idx2word):
                    self.word2idx = word2idx
                    self.idx2word = {int(k): v for k, v in idx2word.items()}  # Convert str keys back to int
                
                def __len__(self):
                    return len(self.word2idx)
            
            return Vocabulary(vocab_dict["word2idx"], vocab_dict["idx2word"])

def predict(model, data_loader, device):
    """
    Make predictions with a trained model.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for prediction data
        device: Device to run predictions on
        
    Returns:
        Tuple of (predictions, true_labels, probabilities)
    """
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            # Handle different types of inputs based on model architecture
            if hasattr(batch, 'keys') and 'input_ids' in batch and 'attention_mask' in batch:
                # For MLM-based models
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                # For LSTM-CNN models
                indices = batch['indices'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                logits = model(indices)
            
            # Get probabilities and predictions
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Store results
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())  # Probabilities for the positive class
    
    return np.array(predictions), np.array(true_labels), np.array(probabilities)

def predict_single_text(model, text, tokenizer=None, vocab=None, device=None, max_length=1024, 
                       caller_only=False, model_type='mlm'):
    """
    Make prediction for a single text input.
    
    Args:
        model: Trained PyTorch model
        text: Input text for prediction
        tokenizer: Tokenizer for MLM models
        vocab: Vocabulary for LSTM-CNN models
        device: Device to run prediction on
        max_length: Maximum sequence length
        caller_only: Whether to use only the caller's utterances
        model_type: Type of model ('mlm' or 'lstm-cnn')
        
    Returns:
        Tuple of (prediction, probability)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    # Extract caller utterances if needed
    if caller_only:
        lines = text.split('\n')
        caller_lines = [line for line in lines if line.startswith('Caller:')]
        text = ' '.join(caller_lines)
    
    # Process input based on model type
    if model_type.lower() == 'mlm':
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for MLM models.")
        
        # Tokenize input
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get prediction
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
    
    elif model_type.lower() == 'lstm-cnn':
        if vocab is None:
            raise ValueError("Vocabulary must be provided for LSTM-CNN models.")
        
        # Check if using RoBERTa tokenizer
        if hasattr(vocab, 'tokenize'):
            # Using RoBERTa tokenizer
            indices = vocab.tokenize(text)
            
            # Pad or truncate sequence
            if len(indices) < max_length:
                indices += [vocab.tokenizer.pad_token_id] * (max_length - len(indices))
            else:
                indices = indices[:max_length]
        else:
            # Using custom vocabulary
            tokens = word_tokenize(text.lower())
            indices = [vocab.word2idx.get(token, 1) for token in tokens]  # 1 is <UNK>
            
            # Pad or truncate sequence
            if len(indices) < max_length:
                indices += [0] * (max_length - len(indices))  # 0 is <PAD>
            else:
                indices = indices[:max_length]
        
        # Convert to tensor and move to device
        indices_tensor = torch.tensor([indices], dtype=torch.long).to(device)
        
        # Get prediction
        with torch.no_grad():
            logits = model(indices_tensor)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Get probabilities and prediction
    probs = F.softmax(logits, dim=1)
    pred = torch.argmax(logits, dim=1).item()
    prob = probs[0, 1].item()  # Probability for the positive class
    
    return pred, prob

def generate_complaint_report(model, text, tokenizer=None, vocab=None, device=None, 
                             max_length=1024, window_size=4, stride=2, threshold=0.5,
                             caller_only=False, model_type='mlm'):
    """
    Generate a complaint report for a full conversation.
    
    Args:
        model: Trained PyTorch model
        text: Full conversation text
        tokenizer: Tokenizer for MLM models
        vocab: Vocabulary for LSTM-CNN models
        device: Device to run prediction on
        max_length: Maximum sequence length
        window_size: Number of utterances to consider in each chunk
        stride: Number of utterances to stride when creating chunks
        threshold: Probability threshold for complaint detection
        caller_only: Whether to use only the caller's utterances
        model_type: Type of model ('mlm' or 'lstm-cnn')
        
    Returns:
        Dictionary with complaint analysis
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split conversation into utterances
    utterances = text.strip().split('\n')
    
    # Create chunks of utterances
    chunks = []
    chunk_indices = []
    
    for i in range(0, len(utterances), stride):
        if i + window_size <= len(utterances):
            chunk = '\n'.join(utterances[i:i + window_size])
            chunks.append(chunk)
            chunk_indices.append((i, i + window_size - 1))
    
    # Make predictions for each chunk
    chunk_preds = []
    chunk_probs = []
    
    for chunk in chunks:
        pred, prob = predict_single_text(
            model, chunk, tokenizer, vocab, device, max_length, caller_only, model_type
        )
        chunk_preds.append(pred)
        chunk_probs.append(prob)
    
    # Analyze results
    complaint_chunks = [i for i, pred in enumerate(chunk_preds) if pred == 1]
    complaint_probs = [chunk_probs[i] for i in complaint_chunks]
    
    complaint_indices = [chunk_indices[i] for i in complaint_chunks]
    
    # Calculate overall complaint percentage
    complaint_percentage = (len(complaint_chunks) / len(chunks)) * 100 if chunks else 0
    
    # Find highest intensity complaint (based on probability)
    highest_intensity_index = np.argmax(chunk_probs) if chunk_probs else None
    highest_intensity_prob = max(chunk_probs) if chunk_probs else 0
    
    # Build report
    report = {
        'complaint_detected': any(pred == 1 for pred in chunk_preds),
        'complaint_percentage': complaint_percentage,
        'total_chunks': len(chunks),
        'complaint_chunks': len(complaint_chunks),
        'complaint_chunk_indices': complaint_indices,
        'complaint_probabilities': complaint_probs,
        'highest_intensity': {
            'chunk_index': highest_intensity_index,
            'probability': highest_intensity_prob,
            'chunk_indices': chunk_indices[highest_intensity_index] if highest_intensity_index is not None else None
        },
        'threshold_used': threshold
    }
    
    return report 