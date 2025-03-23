import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import os
import json
from datetime import datetime

# Import project modules
from data.preprocessor import load_and_preprocess_data, prepare_dataloaders, get_class_weights, HuggingFaceVocab
from models.lstm_cnn_model import LSTMCNN
from models.mlm_model import get_mlm_model_and_tokenizer
from utils.training import train_model, evaluate, measure_inference_latency, measure_inference_latency_cpu, measure_inference_latency_gpu
from utils.inference import predict
from evaluation.visualizations import (
    plot_training_history, 
    plot_confusion_matrix, 
    plot_roc_curve, 
    plot_precision_recall_curve, 
    compare_models, 
    plot_latency_comparison, 
    generate_model_comparison_table
)

def train_lstm_cnn_model(data_dict, model_dir, caller_only=False, device=None, max_epochs=15, use_roberta_tokenizer=False):
    """Train LSTM-CNN model."""
    
    model_type = 'lstm-cnn'
    tokenizer_type = 'roberta' if use_roberta_tokenizer else 'custom'
    model_name = f"lstm_cnn_{tokenizer_type}_{'caller_only' if caller_only else 'full'}"
    print(f"Training {model_name} model...")
    
    # Prepare dataloaders
    dataloaders = prepare_dataloaders(
        data_dict, 
        model_type=model_type, 
        batch_size=32, 
        caller_only=caller_only,
        use_roberta_tokenizer=use_roberta_tokenizer
    )
    
    # Get vocabulary size
    vocab_size = len(dataloaders['vocab'])
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model
    model = LSTMCNN(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_dim=256,
        num_layers=2,
        cnn_out_channels=100,
        kernel_sizes=[3, 5, 7],
        dropout=0.5,
        padding_idx=0 if not use_roberta_tokenizer else dataloaders['vocab'].tokenizer.pad_token_id
    )
    
    # Move model to device
    model = model.to(device)
    
    # Remove class weights - no longer using them for balanced learning
    
    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()  # Removed class weights
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)  # Changed from Adam to AdamW
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Create model save path
    model_save_path = os.path.join(model_dir, f"{model_name}.pt")
    
    # Train model
    model, history = train_model(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=max_epochs,
        early_stopping_patience=5,
        model_save_path=model_save_path
    )
    
    # Evaluate on test set
    test_metrics = evaluate(model, dataloaders['test'], criterion, device)
    print(f"Test metrics for {model_name}:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    
    # Measure CPU inference latency
    print(f"Measuring CPU inference latency for {model_name}...")
    cpu_latency_metrics = measure_inference_latency_cpu(model, dataloaders['test'])
    print(f"CPU inference latency for {model_name}:")
    print(f"Average: {cpu_latency_metrics['avg_latency'] * 1000:.2f} ms")
    print(f"95th percentile: {cpu_latency_metrics['p95_latency'] * 1000:.2f} ms")
    
    # Measure GPU inference latency if available
    gpu_latency_metrics = None
    if torch.cuda.is_available():
        print(f"Measuring GPU inference latency for {model_name}...")
        gpu_latency_metrics = measure_inference_latency_gpu(model, dataloaders['test'])
        print(f"GPU inference latency for {model_name}:")
        print(f"Average: {gpu_latency_metrics['avg_latency'] * 1000:.2f} ms")
        print(f"95th percentile: {gpu_latency_metrics['p95_latency'] * 1000:.2f} ms")
    
    # Plot training history
    history_save_path = os.path.join(model_dir, f"{model_name}_history.png")
    plot_training_history(history, save_path=history_save_path)
    
    # Make predictions and plot evaluation metrics
    predictions, true_labels, probabilities = predict(model, dataloaders['test'], device)
    
    # Plot confusion matrix
    cm_save_path = os.path.join(model_dir, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(true_labels, predictions, save_path=cm_save_path)
    
    # Plot ROC curve
    roc_save_path = os.path.join(model_dir, f"{model_name}_roc_curve.png")
    plot_roc_curve(true_labels, probabilities, save_path=roc_save_path)
    
    # Plot precision-recall curve
    pr_save_path = os.path.join(model_dir, f"{model_name}_pr_curve.png")
    plot_precision_recall_curve(true_labels, probabilities, save_path=pr_save_path)
    
    # Save vocabulary for inference
    if use_roberta_tokenizer:
        # For RoBERTa tokenizer, we just save the tokenizer name
        with open(os.path.join(model_dir, f"{model_name}_vocab.json"), 'w') as f:
            json.dump({"tokenizer_name": "roberta-base"}, f)
    else:
        # For custom vocabulary, save the word2idx and idx2word mappings
        vocab_save_path = os.path.join(model_dir, f"{model_name}_vocab.json")
        vocab_dict = {
            "word2idx": dataloaders['vocab'].word2idx,
            "idx2word": {str(k): v for k, v in dataloaders['vocab'].idx2word.items()},  # Convert int keys to str for JSON
        }
        with open(vocab_save_path, 'w') as f:
            json.dump(vocab_dict, f)
        print(f"Vocabulary saved to {vocab_save_path}")
    
    # Save model metadata
    metadata = {
        'model_type': 'lstm_cnn',
        'tokenizer_type': tokenizer_type,
        'caller_only': caller_only,
        'vocab_size': vocab_size,
        'embedding_dim': 300,
        'hidden_dim': 256,
        'num_layers': 2,
        'cnn_out_channels': 100,
        'kernel_sizes': [3, 5, 7],
        'dropout': 0.5,
        'training_history': {k: [float(v) for v in vs] for k, vs in history.items()},
        'test_metrics': {k: float(v) if isinstance(v, (int, float, bool)) else v.tolist() if hasattr(v, 'tolist') else v 
                       for k, v in test_metrics.items()},
        'cpu_latency_metrics': {
            'avg_latency': float(cpu_latency_metrics['avg_latency']),
            'p95_latency': float(cpu_latency_metrics['p95_latency']),
        }
    }
    
    if gpu_latency_metrics is not None:
        metadata['gpu_latency_metrics'] = {
            'avg_latency': float(gpu_latency_metrics['avg_latency']),
            'p95_latency': float(gpu_latency_metrics['p95_latency']),
        }
    
    metadata_save_path = os.path.join(model_dir, f"{model_name}_metadata.json")
    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        'model': model,
        'test_metrics': test_metrics,
        'latency_metrics': cpu_latency_metrics,
        'gpu_latency_metrics': gpu_latency_metrics,
        'history': history,
        'vocab': dataloaders['vocab']
    }

def train_mlm_model(data_dict, model_dir, caller_only=False, device=None, max_epochs=10, model_name_or_path="roberta-base"):
    """Train MLM model."""
    
    model_name = f"mlm_{'caller_only' if caller_only else 'full'}"
    print(f"Training {model_name} model...")
    
    # Get model and tokenizer
    model, tokenizer = get_mlm_model_and_tokenizer(model_name_or_path)
    
    # Move model to device
    model = model.to(device)
    
    # Prepare dataloaders
    dataloaders = prepare_dataloaders(
        data_dict, 
        tokenizer=tokenizer, 
        model_type='mlm', 
        batch_size=16,  # Smaller batch size for transformer models
        max_length=512,  # Shorter sequences for transformer models
        caller_only=caller_only
    )
    
    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    
    # Create model save path
    model_save_path = os.path.join(model_dir, f"{model_name}.pt")
    
    # Train model
    model, history = train_model(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=max_epochs,
        early_stopping_patience=3,
        model_save_path=model_save_path
    )
    
    # Evaluate on test set
    test_metrics = evaluate(model, dataloaders['test'], criterion, device)
    print(f"Test metrics for {model_name}:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    
    # Measure CPU inference latency
    print(f"Measuring CPU inference latency for {model_name}...")
    cpu_latency_metrics = measure_inference_latency_cpu(model, dataloaders['test'])
    print(f"CPU inference latency for {model_name}:")
    print(f"Average: {cpu_latency_metrics['avg_latency'] * 1000:.2f} ms")
    print(f"95th percentile: {cpu_latency_metrics['p95_latency'] * 1000:.2f} ms")
    
    # Measure GPU inference latency if available
    gpu_latency_metrics = None
    if torch.cuda.is_available():
        print(f"Measuring GPU inference latency for {model_name}...")
        gpu_latency_metrics = measure_inference_latency_gpu(model, dataloaders['test'])
        print(f"GPU inference latency for {model_name}:")
        print(f"Average: {gpu_latency_metrics['avg_latency'] * 1000:.2f} ms")
        print(f"95th percentile: {gpu_latency_metrics['p95_latency'] * 1000:.2f} ms")
    
    # Plot training history
    history_save_path = os.path.join(model_dir, f"{model_name}_history.png")
    plot_training_history(history, save_path=history_save_path)
    
    # Make predictions and plot evaluation metrics
    predictions, true_labels, probabilities = predict(model, dataloaders['test'], device)
    
    # Plot confusion matrix
    cm_save_path = os.path.join(model_dir, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(true_labels, predictions, save_path=cm_save_path)
    
    # Plot ROC curve
    roc_save_path = os.path.join(model_dir, f"{model_name}_roc_curve.png")
    plot_roc_curve(true_labels, probabilities, save_path=roc_save_path)
    
    # Plot precision-recall curve
    pr_save_path = os.path.join(model_dir, f"{model_name}_pr_curve.png")
    plot_precision_recall_curve(true_labels, probabilities, save_path=pr_save_path)
    
    # Save model metadata
    metadata = {
        'model_type': 'mlm',
        'caller_only': caller_only,
        'model_name_or_path': model_name_or_path,
        'training_history': {k: [float(v) for v in vs] for k, vs in history.items()},
        'test_metrics': {k: float(v) if isinstance(v, (int, float, bool)) else v.tolist() if hasattr(v, 'tolist') else v 
                       for k, v in test_metrics.items()},
        'cpu_latency_metrics': {
            'avg_latency': float(cpu_latency_metrics['avg_latency']),
            'p95_latency': float(cpu_latency_metrics['p95_latency']),
        }
    }
    
    if gpu_latency_metrics is not None:
        metadata['gpu_latency_metrics'] = {
            'avg_latency': float(gpu_latency_metrics['avg_latency']),
            'p95_latency': float(gpu_latency_metrics['p95_latency']),
        }
    
    metadata_save_path = os.path.join(model_dir, f"{model_name}_metadata.json")
    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        'model': model,
        'test_metrics': test_metrics,
        'latency_metrics': cpu_latency_metrics,
        'gpu_latency_metrics': gpu_latency_metrics,
        'history': history,
        'tokenizer': tokenizer
    }

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate complaint detection models')
    parser.add_argument('--data_file', type=str, required=True, help='Path to data file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model and results')
    parser.add_argument('--models', type=str, default='all', 
                        choices=['all', 'lstm_cnn_custom_full', 'lstm_cnn_custom_caller_only', 
                                'lstm_cnn_roberta_full', 'lstm_cnn_roberta_caller_only',
                                'mlm_full', 'mlm_caller_only'], 
                        help='Models to train')
    parser.add_argument('--max_rows', type=int, default=None, help='Maximum number of rows to use')
    parser.add_argument('--max_epochs', type=int, default=None, help='Maximum number of epochs')
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(args.output_dir, f"models_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_dict = load_and_preprocess_data(args.data_file, max_rows=args.max_rows)
    
    # Set epochs for each model type if provided
    lstm_cnn_epochs = args.max_epochs if args.max_epochs is not None else 15
    mlm_epochs = args.max_epochs if args.max_epochs is not None else 10
    
    # Dictionary to store model results
    models_results = {}
    
    # Train LSTM-CNN with custom vocab (full conversation)
    if args.models in ['all', 'lstm_cnn_custom_full']:
        lstm_cnn_custom_full_results = train_lstm_cnn_model(
            data_dict, model_dir, caller_only=False, device=device, 
            max_epochs=lstm_cnn_epochs, use_roberta_tokenizer=False
        )
        models_results['LSTM-CNN Custom Vocab (Full)'] = lstm_cnn_custom_full_results
    
    # Train LSTM-CNN with custom vocab (caller-only)
    if args.models in ['all', 'lstm_cnn_custom_caller_only']:
        lstm_cnn_custom_caller_results = train_lstm_cnn_model(
            data_dict, model_dir, caller_only=True, device=device, 
            max_epochs=lstm_cnn_epochs, use_roberta_tokenizer=False
        )
        models_results['LSTM-CNN Custom Vocab (Caller-Only)'] = lstm_cnn_custom_caller_results
    
    # Train LSTM-CNN with RoBERTa tokenizer (full conversation)
    if args.models in ['all', 'lstm_cnn_roberta_full']:
        lstm_cnn_roberta_full_results = train_lstm_cnn_model(
            data_dict, model_dir, caller_only=False, device=device, 
            max_epochs=lstm_cnn_epochs, use_roberta_tokenizer=True
        )
        models_results['LSTM-CNN RoBERTa (Full)'] = lstm_cnn_roberta_full_results
    
    # Train LSTM-CNN with RoBERTa tokenizer (caller-only)
    if args.models in ['all', 'lstm_cnn_roberta_caller_only']:
        lstm_cnn_roberta_caller_results = train_lstm_cnn_model(
            data_dict, model_dir, caller_only=True, device=device, 
            max_epochs=lstm_cnn_epochs, use_roberta_tokenizer=True
        )
        models_results['LSTM-CNN RoBERTa (Caller-Only)'] = lstm_cnn_roberta_caller_results
    
    # Train MLM full conversation model
    if args.models in ['all', 'mlm_full']:
        mlm_full_results = train_mlm_model(
            data_dict, model_dir, caller_only=False, device=device, 
            max_epochs=mlm_epochs
        )
        models_results['MLM (Full)'] = mlm_full_results
    
    # Train MLM caller-only model
    if args.models in ['all', 'mlm_caller_only']:
        mlm_caller_results = train_mlm_model(
            data_dict, model_dir, caller_only=True, device=device, 
            max_epochs=mlm_epochs
        )
        models_results['MLM (Caller-Only)'] = mlm_caller_results
    
    # Compare models if more than one model was trained
    if len(models_results) > 1:
        # Extract test metrics for comparison
        test_metrics = {name: results['test_metrics'] for name, results in models_results.items()}
        
        # Extract latency metrics for comparison
        latency_metrics = {name: results['latency_metrics'] for name, results in models_results.items()}
        
        # Compare models visually
        metrics_save_path = os.path.join(model_dir, "model_comparison.png")
        compare_models(test_metrics, save_path=metrics_save_path)
        
        # Compare latency
        latency_save_path = os.path.join(model_dir, "latency_comparison.png")
        plot_latency_comparison(latency_metrics, save_path=latency_save_path)
        
        # Generate comparison table
        table = generate_model_comparison_table(test_metrics, latency_metrics)
        # Save the table as CSV
        table_save_path = os.path.join(model_dir, "model_comparison_table.csv")
        table.to_csv(table_save_path, index=False)
        
        # Also save as markdown for easy viewing
        markdown_save_path = os.path.join(model_dir, "model_comparison.md")
        with open(markdown_save_path, 'w') as f:
            f.write("# Model Comparison\n\n")
            f.write("## Test Metrics\n\n")
            f.write("| Model | Accuracy | Precision | Recall | F1 Score |\n")
            f.write("|-------|----------|-----------|--------|----------|\n")
            for name, metrics in test_metrics.items():
                f.write(f"| {name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |\n")
            
            f.write("\n## Latency Metrics (ms)\n\n")
            f.write("| Model | Average | 95th Percentile |\n")
            f.write("|-------|---------|----------------|\n")
            for name, metrics in latency_metrics.items():
                f.write(f"| {name} | {metrics['avg_latency'] * 1000:.2f} | {metrics['p95_latency'] * 1000:.2f} |\n")
    
    print(f"All models and results saved to {model_dir}")

if __name__ == "__main__":
    main() 