import torch
import argparse
import os
import json
import glob
import numpy as np
from transformers import AutoTokenizer
import time
import nltk

from src.models.lstm_cnn_model import LSTMCNN
from src.models.mlm_model import MLMClassifier
from src.utils.inference import predict_single_text, generate_complaint_report
from src.data.preprocessor import VocabBuilder

# Download nltk data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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
    if 'lstm_cnn' in model_type.lower():
        model = LSTMCNN(
            vocab_size=metadata['vocab_size'],
            embedding_dim=metadata['embedding_dim'],
            hidden_dim=metadata['hidden_dim'],
            num_layers=metadata['num_layers'],
            cnn_out_channels=metadata['cnn_out_channels'],
            kernel_sizes=metadata['kernel_sizes'],
            dropout=metadata['dropout']
        )
    elif 'mlm' in model_type.lower():
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
    if 'mlm' in model_type.lower():
        # Since we didn't save the tokenizer, get it from the metadata
        metadata_path = model_path.replace('.pt', '_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(metadata['pretrained_model'])
        return tokenizer, None
    elif 'lstm_cnn' in model_type.lower():
        # Load vocabulary
        vocab_path = model_path.replace('.pt', '_vocab.json')
        if os.path.exists(vocab_path):
            print(f"Loading vocabulary from {vocab_path}")
            with open(vocab_path, 'r') as f:
                vocab_dict = json.load(f)
            
            vocab = VocabBuilder()
            # The vocabulary is saved with word2idx and potentially idx2word
            if isinstance(vocab_dict, dict) and "word2idx" in vocab_dict:
                vocab.word2idx = vocab_dict["word2idx"]
                print(f"Loaded word2idx with {len(vocab.word2idx)} entries")
                
                if "idx2word" in vocab_dict:
                    vocab.idx2word = {int(k): v for k, v in vocab_dict["idx2word"].items()}
                    print(f"Loaded idx2word with {len(vocab.idx2word)} entries")
                else:
                    # If idx2word is not saved, create it from word2idx
                    vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
                    print(f"Created idx2word with {len(vocab.idx2word)} entries from word2idx")
            else:
                # Handle the case where vocab_dict is just the word2idx mapping directly
                vocab.word2idx = vocab_dict
                print(f"Loaded flat word2idx with {len(vocab.word2idx)} entries")
                vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
                print(f"Created idx2word with {len(vocab.idx2word)} entries from flat word2idx")
            
            return None, vocab
        else:
            print(f"Vocabulary file not found for {model_path}. Using basic vocabulary.")
            vocab = VocabBuilder()
            vocab.word2idx = {"<PAD>": 0, "<UNK>": 1}
            vocab.idx2word = {0: "<PAD>", 1: "<UNK>"}
            return None, vocab
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def custom_generate_complaint_report(model, text, tokenizer=None, vocab=None, device=None, 
                             max_length=1024, window_size=4, stride=2, threshold=0.5,
                             caller_only=False, model_type='mlm'):
    """A custom implementation of generate_complaint_report with explicit debugging."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Processing conversation with model_type={model_type}, caller_only={caller_only}")
    
    # Split conversation into utterances
    utterances = text.strip().split('\n')
    print(f"Found {len(utterances)} utterances in conversation")
    
    # Create chunks of utterances
    chunks = []
    chunk_indices = []
    
    for i in range(0, len(utterances), stride):
        if i + window_size <= len(utterances):
            chunk = '\n'.join(utterances[i:i + window_size])
            chunks.append(chunk)
            chunk_indices.append((i, i + window_size - 1))
    
    print(f"Created {len(chunks)} chunks for analysis")
    
    # Make predictions for each chunk
    chunk_preds = []
    chunk_probs = []
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        try:
            # Extract caller utterances if needed
            if caller_only:
                chunk_lines = chunk.split('\n')
                caller_lines = [line for line in chunk_lines if line.lower().startswith('caller:')]
                if not caller_lines:
                    print(f"  No caller lines in chunk {i+1}, skipping")
                    chunk_preds.append(0)
                    chunk_probs.append(0.0)
                    continue
                chunk = ' '.join(caller_lines)
                print(f"  Extracted {len(caller_lines)} caller lines")
            
            # Process text based on model type
            if model_type.lower() == 'mlm':
                if tokenizer is None:
                    raise ValueError("Tokenizer must be provided for MLM models.")
                
                # Tokenize input
                encoding = tokenizer(
                    chunk,
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
                
                # Tokenize and convert to indices
                from nltk.tokenize import word_tokenize
                tokens = word_tokenize(chunk.lower())
                print(f"  Tokenized into {len(tokens)} tokens")
                
                indices = [vocab.word2idx.get(token, 1) for token in tokens]  # 1 is <UNK>
                print(f"  Converted to {len(indices)} indices")
                
                # Print sample of tokens and indices for debugging
                if tokens:
                    sample_size = min(5, len(tokens))
                    sample_tokens = tokens[:sample_size]
                    sample_indices = indices[:sample_size]
                    print(f"  Sample tokens: {sample_tokens}")
                    print(f"  Sample indices: {sample_indices}")
                
                # Pad or truncate sequence
                if len(indices) < max_length:
                    indices += [0] * (max_length - len(indices))  # 0 is <PAD>
                else:
                    indices = indices[:max_length]
                
                # Convert to tensor and move to device
                indices_tensor = torch.tensor([indices], dtype=torch.long).to(device)
                print(f"  Created tensor of shape {indices_tensor.shape}")
                
                # Get prediction
                with torch.no_grad():
                    logits = model(indices_tensor)
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Get probabilities and prediction
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()
            prob = probs[0, 1].item()  # Probability for the positive class
            
            chunk_preds.append(pred)
            chunk_probs.append(prob)
            print(f"  Prediction: {pred}, Probability: {prob:.4f}")
            
        except Exception as e:
            import traceback
            print(f"  Error processing chunk {i+1}: {str(e)}")
            traceback.print_exc()
            chunk_preds.append(0)
            chunk_probs.append(0.0)
    
    # Analyze results
    complaint_chunks = [i for i, pred in enumerate(chunk_preds) if pred == 1]
    complaint_probs = [chunk_probs[i] for i in complaint_chunks]
    
    complaint_indices = [chunk_indices[i] for i in complaint_chunks]
    
    # Calculate overall complaint percentage
    complaint_percentage = (len(complaint_chunks) / len(chunks)) * 100 if chunks else 0
    
    # Find highest intensity chunk (based on probability)
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
        'chunk_probs': chunk_probs,  # Add all chunk probabilities
        'chunk_preds': chunk_preds,  # Add all chunk predictions
        'highest_intensity': {
            'chunk_index': highest_intensity_index,
            'probability': highest_intensity_prob,
            'chunk_indices': chunk_indices[highest_intensity_index] if highest_intensity_index is not None else None
        },
        'threshold_used': threshold
    }
    
    return report

def process_conversation(conversation, model, tokenizer=None, vocab=None, device=None, 
                        caller_only=False, model_type='mlm'):
    """Process a conversation and generate a complaint report."""
    try:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use our custom implementation for more detailed debugging
        print(f"\nProcessing conversation using {model_type} model with caller_only={caller_only}")
        
        # Generate report using our custom implementation
        report = custom_generate_complaint_report(
            model, conversation, tokenizer, vocab, device, 
            caller_only=caller_only, model_type=model_type
        )
        
        return report
    except Exception as e:
        # Print detailed error info
        import traceback
        print(f"Error generating report: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        
        # Print additional debugging info
        if vocab:
            print(f"Vocabulary size: {len(vocab.word2idx)}")
            print(f"First 5 vocab items: {list(vocab.word2idx.items())[:5]}")
        
        return {
            'complaint_detected': False,
            'complaint_percentage': 0,
            'total_chunks': 0,
            'complaint_chunks': 0,
            'complaint_probabilities': []
        }

def print_report(model_name, report):
    """Print a formatted complaint report."""
    print("\n" + "="*50)
    print(f"Model: {model_name}")
    print("Complaint Detection Report")
    print("="*50)
    
    print(f"Complaint Detected: {'Yes' if report['complaint_detected'] else 'No'}")
    
    # Show probabilities even if no complaints were detected
    if 'complaint_probabilities' in report and len(report['complaint_probabilities']) > 0:
        print(f"Complaint Probability: {report['complaint_probabilities'][0]:.4f}")
    elif 'highest_intensity' in report and 'probability' in report['highest_intensity']:
        print(f"Highest Probability: {report['highest_intensity']['probability']:.4f}")
    else:
        print("No probability data available")
    
    # Show chunk predictions
    if 'chunk_probs' in report and len(report['chunk_probs']) > 0:
        print("\nChunk Probabilities:")
        for i, prob in enumerate(report['chunk_probs']):
            print(f"  Chunk {i+1}: {prob:.4f}")
    
    print("="*50)

def get_sample_conversations():
    """Return sample conversations for testing."""
    conversations = [
        # Sample 1: Complaint about service
        """Caller: Hello, I've been trying to get my internet fixed for a week now and nobody seems to care.
Agent: I'm sorry to hear that. Let me look into this for you.
Caller: I've called three times already and each time I was promised someone would come, but nobody ever showed up.
Agent: I apologize for the inconvenience. I can see the notes on your account.
Caller: This is ridiculous! I'm paying for a service I'm not receiving.
Agent: I understand your frustration. Let me schedule a technician visit with our highest priority.
Caller: I want a refund for the days I haven't had service.
Agent: That's a reasonable request. I'll process a credit for the days affected.""",
        
        # Sample 2: No complaint, regular service call
        """Caller: Hi, I'm calling to upgrade my plan.
Agent: Hello! I'd be happy to help you with that. What plan are you interested in?
Caller: I saw the premium package online. It has more channels.
Agent: The premium package is a great choice. It includes 200+ channels, including HBO and Showtime.
Caller: That sounds good. How much would it cost?
Agent: The premium package is $89.99 per month. I can also offer a 3-month discount at $69.99 if you upgrade today.
Caller: That sounds like a good deal. Let's go with that.
Agent: Excellent! I'll process that upgrade right away for you."""
    ]
    return conversations

def main():
    parser = argparse.ArgumentParser(description='Test all trained models with sample conversations')
    parser.add_argument('--models_dir', type=str, required=True, help='Directory containing trained models')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get all model files
    model_files = glob.glob(os.path.join(args.models_dir, '*.pt'))
    if not model_files:
        print(f"No model files found in {args.models_dir}")
        return
    
    print(f"Found {len(model_files)} models in {args.models_dir}")
    
    # Get sample conversations
    conversations = get_sample_conversations()
    
    # Process each model with each conversation
    for model_path in model_files:
        model_name = os.path.basename(model_path).replace('.pt', '')
        model_type = model_name
        
        caller_only = 'caller_only' in model_name
        
        try:
            # Load model
            print(f"\nLoading model: {model_name}")
            model, metadata = load_model(model_path, model_type, device)
            
            # Load tokenizer or vocabulary
            tokenizer, vocab = load_tokenizer_or_vocab(model_path, model_type)
            
            # Process each conversation
            for i, conversation in enumerate(conversations):
                print(f"\n--- Sample Conversation {i+1} ---")
                print("First few lines of conversation:")
                print('\n'.join(conversation.split('\n')[:3]) + "...")
                
                # Process conversation
                start_time = time.time()
                report = process_conversation(
                    conversation, model, tokenizer, vocab, device, 
                    caller_only=caller_only, 
                    model_type='mlm' if 'mlm' in model_type else 'lstm-cnn'
                )
                inference_time = time.time() - start_time
                
                # Add inference time to report
                report['inference_time'] = inference_time
                
                # Print report
                print_report(model_name, report)
                print(f"Inference time: {inference_time:.4f} seconds")
                
        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            continue

if __name__ == '__main__':
    main() 