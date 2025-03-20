import torch
import argparse
import os
import json
from transformers import AutoTokenizer

from models.lstm_cnn_model import LSTMCNN
from models.mlm_model import MLMClassifier
from utils.inference import predict_single_text, generate_complaint_report
from data.preprocessor import VocabBuilder

def load_model(model_path, model_type, device=None):
    """Load a trained model."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
            model_name=metadata['pretrained_model']
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, metadata

def load_tokenizer_or_vocab(model_path, model_type):
    """Load the tokenizer or vocabulary for a trained model."""
    if model_type.lower() == 'mlm':
        # Load tokenizer
        tokenizer_path = model_path.replace('.pt', '_tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return tokenizer, None
    elif model_type.lower() == 'lstm-cnn':
        # Load vocabulary if available
        vocab_path = model_path.replace('.pt', '_vocab.json')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocab_dict = json.load(f)
            
            vocab = VocabBuilder()
            vocab.word2idx = {k: int(v) for k, v in vocab_dict.items()}
            vocab.idx2word = {int(k): v for v, k in vocab.word2idx.items()}
            
            return None, vocab
        else:
            print("Vocabulary file not found. Please provide a vocabulary.")
            return None, None
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def process_conversation(conversation, model, tokenizer=None, vocab=None, device=None, 
                        caller_only=False, model_type='mlm'):
    """Process a conversation and generate a complaint report."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate report
    report = generate_complaint_report(
        model, conversation, tokenizer, vocab, device, 
        caller_only=caller_only, model_type=model_type
    )
    
    return report

def print_report(report):
    """Print a formatted complaint report."""
    print("\n" + "="*50)
    print("Complaint Detection Report")
    print("="*50)
    
    print(f"Complaint Detected: {'Yes' if report['complaint_detected'] else 'No'}")
    print(f"Complaint Percentage: {report['complaint_percentage']:.2f}%")
    print(f"Total Conversation Chunks: {report['total_chunks']}")
    print(f"Complaint Chunks: {report['complaint_chunks']}")
    
    if report['complaint_detected']:
        print("\nHighest Intensity Complaint:")
        print(f"  Probability: {report['highest_intensity']['probability']:.4f}")
        print(f"  Location: Lines {report['highest_intensity']['chunk_indices'][0]}-{report['highest_intensity']['chunk_indices'][1]}")
        
        print("\nAll Complaint Locations:")
        for i, indices in enumerate(report['complaint_chunk_indices']):
            print(f"  Complaint {i+1}: Lines {indices[0]}-{indices[1]} (Prob: {report['complaint_probabilities'][i]:.4f})")
    
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Inference demo for complaint detection models')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file (.pt)')
    parser.add_argument('--model_type', type=str, required=True, choices=['lstm-cnn', 'mlm'], 
                        help='Type of model (lstm-cnn or mlm)')
    parser.add_argument('--input_file', type=str, help='Path to input conversation file')
    parser.add_argument('--caller_only', action='store_true', help='Use only caller utterances for prediction')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading {args.model_type} model from {args.model_path}...")
    model, metadata = load_model(args.model_path, args.model_type, device)
    
    # Load tokenizer or vocabulary
    tokenizer, vocab = load_tokenizer_or_vocab(args.model_path, args.model_type)
    
    # Load conversation from file or get from user input
    if args.input_file:
        with open(args.input_file, 'r') as f:
            conversation = f.read()
    else:
        print("\nEnter conversation (format: 'Agent: ...' for agent utterances and 'Caller: ...' for caller utterances)")
        print("Type 'END' on a new line when finished.\n")
        
        lines = []
        while True:
            line = input()
            if line.strip().upper() == 'END':
                break
            lines.append(line)
        
        conversation = '\n'.join(lines)
    
    # Process conversation
    report = process_conversation(
        conversation, model, tokenizer, vocab, device, 
        caller_only=args.caller_only, 
        model_type=args.model_type
    )
    
    # Print report
    print_report(report)

if __name__ == '__main__':
    main() 