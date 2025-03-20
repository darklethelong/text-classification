import torch
import json
import os
import numpy as np
from nltk.tokenize import word_tokenize

from src.models.lstm_cnn_model import LSTMCNN
from src.data.preprocessor import VocabBuilder

def load_model_and_vocab(model_path):
    """Load a trained model and its vocabulary."""
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model metadata
    metadata_path = model_path.replace('.pt', '_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create model with same parameters
    model = LSTMCNN(
        vocab_size=metadata['vocab_size'],
        embedding_dim=metadata['embedding_dim'],
        hidden_dim=metadata['hidden_dim'],
        num_layers=metadata['num_layers'],
        cnn_out_channels=metadata['cnn_out_channels'],
        kernel_sizes=metadata['kernel_sizes'],
        dropout=metadata['dropout']
    )
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load vocabulary
    vocab_path = model_path.replace('.pt', '_vocab.json')
    with open(vocab_path, 'r') as f:
        vocab_dict = json.load(f)
    
    vocab = VocabBuilder()
    vocab.word2idx = vocab_dict['word2idx']
    vocab.idx2word = {int(k): v for k, v in vocab_dict['idx2word'].items()}
    
    return model, vocab, device

def predict_text(model, vocab, text, device):
    """Make a prediction on a text input."""
    # Tokenize and convert to indices
    tokens = word_tokenize(text.lower())
    indices = [vocab.word2idx.get(token, 1) for token in tokens]  # 1 is <UNK>
    
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
    indices_tensor = torch.tensor([indices], dtype=torch.long).to(device)
    
    # Get prediction
    with torch.no_grad():
        logits = model(indices_tensor)
    
    # Get probabilities and prediction
    probs = torch.nn.functional.softmax(logits, dim=1)
    pred = torch.argmax(logits, dim=1).item()
    prob = probs[0, 1].item()  # Probability for the positive class
    
    return pred, prob

def main():
    # Get latest model path
    model_dir = "output/models_20250320_124018"
    model_path = os.path.join(model_dir, "lstm_cnn_full.pt")
    
    # Load model and vocabulary
    model, vocab, device = load_model_and_vocab(model_path)
    print(f"Loaded model from {model_path}")
    print(f"Vocabulary size: {len(vocab.word2idx)}")
    
    # Define some test conversations
    test_conversations = [
        # Complaint
        """Caller: Hello, I've been trying to get my internet fixed for a week now and nobody seems to care.
Agent: I'm sorry to hear that. Let me look into this for you.
Caller: I've called three times already and each time I was promised someone would come, but nobody ever showed up.
Agent: I apologize for the inconvenience. I can see the notes on your account.
Caller: This is ridiculous! I'm paying for a service I'm not receiving.
Agent: I understand your frustration. Let me schedule a technician visit with our highest priority.
Caller: I want a refund for the days I haven't had service.
Agent: That's a reasonable request. I'll process a credit for the days affected.""",
        
        # Not a complaint
        """Caller: Hi, I'm calling to upgrade my plan.
Agent: Hello! I'd be happy to help you with that. What plan are you interested in?
Caller: I saw the premium package online. It has more channels.
Agent: The premium package is a great choice. It includes 200+ channels, including HBO and Showtime.
Caller: That sounds good. How much would it cost?
Agent: The premium package is $89.99 per month. I can also offer a 3-month discount at $69.99 if you upgrade today.
Caller: That sounds like a good deal. Let's go with that.
Agent: Excellent! I'll process that upgrade right away for you."""
    ]
    
    # Make predictions on test conversations
    print("\nPredicting on test conversations:")
    for i, conversation in enumerate(test_conversations):
        pred, prob = predict_text(model, vocab, conversation, device)
        label = "Complaint" if pred == 1 else "Not Complaint"
        print(f"\nTest Conversation {i+1}:")
        print(f"Prediction: {label} (class {pred})")
        print(f"Probability: {prob:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main() 