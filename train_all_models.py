#!/usr/bin/env python
"""
Train and evaluate all six models for complaint detection:
1. LSTM-CNN with custom vocab (full conversation)
2. LSTM-CNN with custom vocab (caller-only)
3. LSTM-CNN with RoBERTa tokenizer (full conversation)
4. LSTM-CNN with RoBERTa tokenizer (caller-only)
5. MLM (full conversation)
6. MLM (caller-only)
"""

import subprocess
import argparse
import sys
import os
from datetime import datetime

def train_models(data_file, output_dir, max_rows, max_epochs):
    """Train all six models for complaint detection."""
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(output_dir, f"models_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Define command template
    cmd_template = [
        sys.executable,
        "src/train_and_evaluate.py",
        "--data_file", data_file,
        "--output_dir", output_dir
    ]
    
    # Add max_rows if specified
    if max_rows:
        cmd_template.extend(["--max_rows", str(max_rows)])
    
    # Add max_epochs if specified
    if max_epochs:
        cmd_template.extend(["--max_epochs", str(max_epochs)])
    
    # Train all models in sequence
    try:
        print("=== Training all six models ===")
        cmd = cmd_template + ["--models", "all"]
        print(f"Running command: {' '.join(cmd)}")
        
        # Run command with stdout and stderr captured and printed in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Print real-time output
        for line in process.stdout:
            print(line, end='')
        
        # Wait for process to complete
        process.wait()
        
        # Check return code
        if process.returncode != 0:
            # Print any error output
            for line in process.stderr:
                print(line, end='')
            print(f"Command failed with return code {process.returncode}")
            raise subprocess.CalledProcessError(process.returncode, cmd)
        
        print("All models trained successfully!")
        
        return model_dir
    except subprocess.CalledProcessError as e:
        print(f"Error while training models: {e}")
        print("Training each model individually...")
        
        models = [
            "lstm_cnn_custom_full",
            "lstm_cnn_custom_caller_only",
            "lstm_cnn_roberta_full",
            "lstm_cnn_roberta_caller_only",
            "mlm_full",
            "mlm_caller_only"
        ]
        
        successful_models = []
        
        for model in models:
            try:
                print(f"\n=== Training {model} ===")
                cmd = cmd_template + ["--models", model]
                print(f"Running command: {' '.join(cmd)}")
                
                # Run command with stdout and stderr captured and printed in real-time
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1  # Line buffered
                )
                
                # Print real-time output
                for line in process.stdout:
                    print(line, end='')
                
                # Wait for process to complete
                process.wait()
                
                # Check return code
                if process.returncode != 0:
                    # Print any error output
                    for line in process.stderr:
                        print(line, end='')
                    print(f"Command failed with return code {process.returncode}")
                    raise subprocess.CalledProcessError(process.returncode, cmd)
                
                successful_models.append(model)
                print(f"{model} trained successfully!")
            except subprocess.CalledProcessError as e:
                print(f"Error while training {model}: {e}")
        
        if successful_models:
            print(f"\nSuccessfully trained {len(successful_models)} of {len(models)} models:")
            for model in successful_models:
                print(f"  - {model}")
        else:
            print("\nFailed to train any models.")
            
        return model_dir

def run_inference(model_dir, best_model_path=None):
    """Run inference with the best model."""
    if not best_model_path:
        print("Looking for best model in model directory...")
        
        # Look for the highest F1 score in model_comparison.md
        comparison_file = os.path.join(model_dir, "model_comparison.md")
        if not os.path.exists(comparison_file):
            print(f"Model comparison file not found: {comparison_file}")
            return
        
        # Read comparison file
        with open(comparison_file, 'r') as f:
            lines = f.readlines()
        
        # Extract model performance
        models = []
        best_f1 = 0
        best_model_name = None
        
        # Find the table lines
        for i, line in enumerate(lines):
            if line.startswith('| LSTM-CNN') or line.startswith('| MLM'):
                parts = line.split('|')
                if len(parts) >= 6:  # Model | Accuracy | Precision | Recall | F1 Score |
                    model_name = parts[1].strip()
                    try:
                        f1_score = float(parts[-2].strip())
                        models.append((model_name, f1_score))
                        if f1_score > best_f1:
                            best_f1 = f1_score
                            best_model_name = model_name
                    except ValueError:
                        continue
        
        if not best_model_name:
            print("Could not determine the best model.")
            return
        
        print(f"Best model: {best_model_name} (F1 Score: {best_f1:.4f})")
        
        # Convert model name to file pattern
        if "LSTM-CNN Custom Vocab (Full)" in best_model_name:
            best_model_file = "lstm_cnn_custom_full.pt"
        elif "LSTM-CNN Custom Vocab (Caller-Only)" in best_model_name:
            best_model_file = "lstm_cnn_custom_caller_only.pt"
        elif "LSTM-CNN RoBERTa (Full)" in best_model_name:
            best_model_file = "lstm_cnn_roberta_full.pt"
        elif "LSTM-CNN RoBERTa (Caller-Only)" in best_model_name:
            best_model_file = "lstm_cnn_roberta_caller_only.pt"
        elif "MLM (Full)" in best_model_name:
            best_model_file = "mlm_full.pt"
        elif "MLM (Caller-Only)" in best_model_name:
            best_model_file = "mlm_caller_only.pt"
        else:
            print(f"Unknown model name format: {best_model_name}")
            return
        
        best_model_path = os.path.join(model_dir, best_model_file)
    
    # Check if best model exists
    if not os.path.exists(best_model_path):
        print(f"Best model file not found: {best_model_path}")
        return
    
    # Create test inference script
    test_script = "test_inference.py"
    test_script_content = """
import torch
import json
import os
import sys
from src.models.lstm_cnn_model import LSTMCNN
from src.models.mlm_model import MLMClassifier
from src.utils.inference import load_tokenizer_or_vocab, predict_single_text
from transformers import AutoTokenizer

def load_model(model_path, device):
    \"\"\"Load model from path\"\"\"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    if "lstm_cnn" in model_path:
        # Load LSTM-CNN model
        vocab_size = checkpoint['vocab_size']
        model = LSTMCNN(
            vocab_size=vocab_size,
            embedding_dim=300,
            hidden_dim=256,
            num_layers=2,
            cnn_out_channels=100,
            kernel_sizes=[3, 5, 7],
            dropout=0.5,
            padding_idx=0
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model_type = 'lstm-cnn'
    else:
        # Load MLM model
        model = MLMClassifier()
        model.load_state_dict(checkpoint['model_state_dict'])
        model_type = 'mlm'
    
    model.eval()
    return model, model_type

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} MODEL_PATH")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, model_type = load_model(model_path, device)
    model = model.to(device)
    
    # Load vocabulary or tokenizer
    vocab_path = model_path.replace('.pt', '_vocab.json')
    if os.path.exists(vocab_path):
        tokenizer_or_vocab = load_tokenizer_or_vocab(vocab_path, model_type)
        print(f"Loaded vocabulary/tokenizer from {vocab_path}")
    else:
        # For MLM models, try to extract tokenizer name from metadata
        metadata_path = model_path.replace('.pt', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            if 'model_name_or_path' in metadata:
                tokenizer_or_vocab = AutoTokenizer.from_pretrained(metadata['model_name_or_path'])
                print(f"Loaded tokenizer from {metadata['model_name_or_path']}")
            else:
                print("Could not find tokenizer information.")
                sys.exit(1)
        else:
            print(f"Vocabulary/tokenizer file not found: {vocab_path}")
            sys.exit(1)
    
    # Determine caller_only from model path
    caller_only = 'caller_only' in model_path
    
    # Test conversations
    test_conversations = [
        # Complaint
        '''Caller: Hello, I've been trying to get my internet fixed for a week now and nobody seems to care.
Agent: I'm sorry to hear that. Let me look into this for you.
Caller: I've called three times already and each time I was promised someone would come, but nobody ever showed up.
Agent: I apologize for the inconvenience. I can see the notes on your account.''',
        
        # Non-complaint
        '''Caller: Hi, I'm calling to upgrade my plan.
Agent: Hello! I'd be happy to help you with that. What plan are you interested in?
Caller: I saw the premium package online. It has more channels.
Agent: The premium package is a great choice. It includes 200+ channels.'''
    ]
    
    # Make predictions
    print("\\nPredicting on test conversations:\\n")
    for i, conv in enumerate(test_conversations):
        print(f"Test Conversation {i+1}:")
        
        if model_type == 'mlm':
            # For MLM models
            pred, prob = predict_single_text(
                model, 
                conv, 
                tokenizer=tokenizer_or_vocab, 
                device=device, 
                caller_only=caller_only, 
                model_type=model_type
            )
        else:
            # For LSTM-CNN models
            pred, prob = predict_single_text(
                model, 
                conv, 
                vocab=tokenizer_or_vocab, 
                device=device, 
                caller_only=caller_only, 
                model_type=model_type
            )
        
        pred_class = "Complaint" if pred == 1 else "Non-complaint"
        print(f"Prediction: {pred_class} (class {pred})")
        print(f"Probability: {prob:.4f}")
        print(f"{'--' * 25}\\n")

if __name__ == "__main__":
    main()
"""
    
    with open(test_script, 'w') as f:
        f.write(test_script_content)
    
    print(f"\nRunning inference with best model: {best_model_path}")
    cmd = [sys.executable, test_script, best_model_path]
    try:
        # Run command with stdout and stderr captured and printed in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Print real-time output
        for line in process.stdout:
            print(line, end='')
        
        # Wait for process to complete
        process.wait()
        
        # Check return code
        if process.returncode != 0:
            # Print any error output
            for line in process.stderr:
                print(line, end='')
            print(f"Command failed with return code {process.returncode}")
            raise subprocess.CalledProcessError(process.returncode, cmd)
            
    except subprocess.CalledProcessError as e:
        print(f"Error during inference: {e}")

def main():
    """Main function to parse arguments and run the training and inference."""
    parser = argparse.ArgumentParser(description="Train and evaluate all six models for complaint detection")
    parser.add_argument("--data_file", type=str, default="train.csv", help="Path to data file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save model and results")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum number of rows to use")
    parser.add_argument("--max_epochs", type=int, default=None, help="Maximum number of epochs")
    parser.add_argument("--models", type=str, default="all", help="Models to train (comma-separated)")
    parser.add_argument("--inference_only", action="store_true", help="Run inference only (no training)")
    parser.add_argument("--model_path", type=str, help="Path to specific model for inference")
    
    args = parser.parse_args()
    
    if args.inference_only:
        if args.model_path:
            model_dir = os.path.dirname(args.model_path)
            run_inference(model_dir, args.model_path)
        else:
            print("Please specify --model_path when using --inference_only")
    else:
        model_dir = train_models(args.data_file, args.output_dir, args.max_rows, args.max_epochs)
        run_inference(model_dir, args.model_path)

if __name__ == "__main__":
    main() 