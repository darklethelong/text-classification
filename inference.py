import torch
import os
import json
import argparse
import sys
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer
from prettytable import PrettyTable

from src.models.lstm_cnn_model import LSTMCNN
from src.models.mlm_model import MLMClassifier
from src.data.preprocessor import VocabBuilder
from src.utils.inference import generate_complaint_report

class VocabBuilder:
    """Simple vocabulary builder class for LSTM models."""
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
    
    def __len__(self):
        return len(self.word2idx)

def load_model(model_path, model_type, device=None):
    """Load a trained model."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {model_path}...")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get model metadata
        metadata_path = model_path.replace('.pt', '_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Initialize model based on type
        if model_type.lower() == 'lstm-cnn':
            model = LSTMCNN(
                vocab_size=metadata['vocab_size'],
                embedding_dim=metadata['embedding_dim'],
                hidden_dim=metadata['hidden_dim'],
                num_layers=metadata['num_layers'],
                cnn_out_channels=metadata['cnn_out_channels'],
                kernel_sizes=metadata['kernel_sizes'],
                dropout=metadata['dropout']
            )
        elif model_type.lower() == 'mlm':
            model = MLMClassifier(
                model_name=metadata.get('pretrained_model', "jinaai/jina-embeddings-v2-small-en")
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        return model, metadata
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_tokenizer_or_vocab(model_path, model_type):
    """Load the tokenizer or vocabulary for a trained model."""
    if model_type.lower() == 'mlm':
        # Load tokenizer
        tokenizer_path = model_path.replace('.pt', '_tokenizer')
        if os.path.exists(tokenizer_path):
            print(f"Loading tokenizer from {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            return tokenizer, None
        else:
            # Fallback to default tokenizer
            print(f"Tokenizer not found at {tokenizer_path}, loading default")
            tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-small-en")
            return tokenizer, None
    
    elif model_type.lower() == 'lstm-cnn':
        # For LSTM-CNN, we need the vocabulary from the saved vocab file
        vocab_path = model_path.replace('.pt', '_vocab.json')
        
        if os.path.exists(vocab_path):
            print(f"Loading vocabulary from {vocab_path}")
            with open(vocab_path, 'r') as f:
                vocab_dict = json.load(f)
            
            # Create vocabulary object
            vocab = VocabBuilder()
            
            # Load word2idx and idx2word
            if 'word2idx' in vocab_dict:
                vocab.word2idx = vocab_dict['word2idx']
            
            if 'idx2word' in vocab_dict:
                # Convert string keys back to integers
                vocab.idx2word = {int(k): v for k, v in vocab_dict['idx2word'].items()}
            
            print(f"Loaded vocabulary with {len(vocab.word2idx)} words")
            return None, vocab
        else:
            print(f"WARNING: Vocabulary file not found at {vocab_path}")
            print("Creating a basic vocabulary (inference may not be accurate)")
            
            # Create a basic vocabulary for demonstration
            vocab = VocabBuilder()
            
            # Add some common words for call center conversations
            words = ['agent', 'caller', 'hello', 'yes', 'no', 'thank', 'you', 'please', 'help',
                    'problem', 'issue', 'complaint', 'service', 'account', 'sorry', 'frustrated',
                    'angry', 'disappointed', 'wait', 'payment', 'refund', 'cancel', 'order', 'call',
                    'manager', 'supervisor', 'speak', 'money', 'charged', 'credit', 'card', 'wrong',
                    'error', 'mistake', 'fix', 'resolve', 'solution', 'customer', 'representative']
            
            for i, word in enumerate(words, start=2):  # Start after PAD and UNK
                vocab.word2idx[word] = i
                vocab.idx2word[i] = word
            
            return None, vocab
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def load_all_models(models_dir, device=None):
    """Load all models from the specified directory."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = []
    
    # Find all model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    
    if not model_files:
        print(f"No model files found in {models_dir}")
        return []
    
    print(f"Found {len(model_files)} model files: {model_files}")
    
    for file in model_files:
        model_path = os.path.join(models_dir, file)
        model_name = file.replace('.pt', '')
        
        # Determine model type
        if model_name.startswith('lstm_cnn'):
            model_type = 'lstm-cnn'
        elif model_name.startswith('mlm'):
            model_type = 'mlm'
        else:
            print(f"Unknown model type for {model_name}, skipping")
            continue
        
        # Determine if caller-only
        caller_only = 'caller_only' in model_name
        
        try:
            # Load model and preprocessing tools
            model, metadata = load_model(model_path, model_type, device)
            tokenizer, vocab = load_tokenizer_or_vocab(model_path, model_type)
            
            models.append({
                'name': model_name,
                'type': model_type,
                'model': model,
                'tokenizer': tokenizer,
                'vocab': vocab,
                'caller_only': caller_only,
                'metadata': metadata
            })
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
    
    return models

def test_sample_conversation(models, conversation, device=None):
    """Test all models on a sample conversation."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a table for results
    results_table = PrettyTable()
    results_table.field_names = ["Model", "Prediction", "Probability"]
    
    # Test each model on the conversation
    for model_info in models:
        try:
            # Generate complaint report
            report = generate_complaint_report(
                model_info['model'],
                conversation,
                model_info['tokenizer'],
                model_info['vocab'],
                device,
                caller_only=model_info['caller_only'],
                model_type=model_info['type']
            )
            
            # Add results to table
            results_table.add_row([
                model_info['name'],
                "Complaint" if report['complaint_detected'] else "Non-Complaint",
                f"{report['complaint_percentage']:.2f}%"
            ])
            
            # Print detailed report
            print(f"\n{'-'*50}")
            print(f"Model: {model_info['name']}")
            print(f"Type: {model_info['type']}")
            print(f"Caller-only: {model_info['caller_only']}")
            
            print("\nComplaint Detection Report:")
            print(f"Complaint Detected: {'Yes' if report['complaint_detected'] else 'No'}")
            print(f"Complaint Percentage: {report['complaint_percentage']:.2f}%")
            print(f"Total Conversation Chunks: {report['total_chunks']}")
            print(f"Complaint Chunks: {report['complaint_chunks']}")
            
            if report['complaint_detected'] and report['complaint_chunks'] > 0:
                print("\nHighest Intensity Complaint:")
                print(f"  Probability: {report['highest_intensity']['probability']:.4f}")
                print(f"  Location: Lines {report['highest_intensity']['chunk_indices'][0]}-{report['highest_intensity']['chunk_indices'][1]}")
                
                print("\nAll Complaint Locations:")
                for i, indices in enumerate(report['complaint_chunk_indices']):
                    print(f"  Complaint {i+1}: Lines {indices[0]}-{indices[1]} (Prob: {report['complaint_probabilities'][i]:.4f})")
        
        except Exception as e:
            print(f"Error testing model {model_info['name']}: {str(e)}")
    
    return results_table

def main():
    # Ensure NLTK resources are available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    parser = argparse.ArgumentParser(description='Run inference with trained complaint detection models')
    parser.add_argument('--models_dir', type=str, required=True, help='Directory containing the trained models')
    parser.add_argument('--conversation_file', type=str, help='Path to a file containing the conversation to analyze')
    parser.add_argument('--model_type', type=str, choices=['lstm-cnn', 'mlm'], help='Type of model to use (required if specifying --model_path)')
    parser.add_argument('--model_path', type=str, help='Path to a specific model file')
    parser.add_argument('--caller_only', action='store_true', help='Only consider caller utterances')
    
    args = parser.parse_args()
    
    # Check that we have a valid models directory
    if not os.path.exists(args.models_dir):
        print(f"Error: Models directory {args.models_dir} does not exist")
        sys.exit(1)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    models = []
    
    if args.model_path:
        # Load a specific model
        if not args.model_type:
            print("Error: --model_type must be specified when using --model_path")
            sys.exit(1)
        
        try:
            model_path = args.model_path
            if not os.path.exists(model_path):
                model_path = os.path.join(args.models_dir, args.model_path)
            
            model, metadata = load_model(model_path, args.model_type, device)
            tokenizer, vocab = load_tokenizer_or_vocab(model_path, args.model_type)
            
            models.append({
                'name': os.path.basename(model_path).replace('.pt', ''),
                'type': args.model_type,
                'model': model,
                'tokenizer': tokenizer,
                'vocab': vocab,
                'caller_only': args.caller_only,
                'metadata': metadata
            })
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    else:
        # Load all models in the directory
        models = load_all_models(args.models_dir, device)
        
        if not models:
            print(f"No models could be loaded from {args.models_dir}")
            sys.exit(1)
    
    # Get the conversation to analyze
    if args.conversation_file:
        # Load from file
        if not os.path.exists(args.conversation_file):
            print(f"Error: Conversation file {args.conversation_file} does not exist")
            sys.exit(1)
        
        with open(args.conversation_file, 'r') as f:
            conversation = f.read()
    else:
        # Use a sample conversation
        print("\nNo conversation file provided. Using sample conversations.")
        
        # Sample conversations
        sample_conversations = [
            # Complaint conversation
            """Agent: Thank you for calling our customer service. How may I help you today?
Caller: I've been having issues with my internet service for the past week. It keeps disconnecting.
Agent: I'm sorry to hear that. Let me check your account and see what's going on.
Caller: This is the third time I've called about this issue and nothing has been resolved. I'm really frustrated with the service.
Agent: I understand your frustration. Let me see what I can do to help.
Caller: I expect better service for the amount I pay each month. This is unacceptable.
Agent: You're absolutely right, and I apologize for the inconvenience. Let's get this resolved today.""",
            
            # Non-complaint conversation
            """Agent: Thank you for calling our customer service. How may I help you today?
Caller: I'm calling to check the status of my order.
Agent: I'd be happy to help with that. May I have your order number please?
Caller: Yes, it's AB12345.
Agent: Thank you. Let me check that for you... I can see your order has been shipped and should arrive by Thursday.
Caller: Great, thanks for checking.
Agent: Is there anything else I can help you with today?
Caller: No, that's all I needed. Thanks for your help."""
        ]
        
        # Ask the user to choose a conversation or enter their own
        print("\nChoose a sample conversation or enter your own:")
        print("[1] Complaint conversation")
        print("[2] Non-complaint conversation")
        print("[3] Enter your own conversation")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            conversation = sample_conversations[0]
        elif choice == '2':
            conversation = sample_conversations[1]
        elif choice == '3':
            print("\nEnter your conversation below (format: 'Agent: ...' or 'Caller: ...')")
            print("Type 'END' on a new line when finished.\n")
            
            lines = []
            while True:
                line = input()
                if line.strip().upper() == 'END':
                    break
                lines.append(line)
            
            conversation = '\n'.join(lines)
        else:
            print("Invalid choice. Using complaint conversation by default.")
            conversation = sample_conversations[0]
    
    print("\nConversation to analyze:")
    print("-" * 50)
    print(conversation)
    print("-" * 50)
    
    # Analyze the conversation with all models
    print("\nAnalyzing conversation with all models...")
    results_table = test_sample_conversation(models, conversation, device)
    
    # Print summary results
    print("\nSummary Results:")
    print(results_table)

if __name__ == "__main__":
    main() 