import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import os
import json
from datetime import datetime

# Import project modules
from data.preprocessor import load_and_preprocess_data, prepare_dataloaders, get_class_weights
from models.lstm_cnn_model import LSTMCNN
from models.mlm_model import get_mlm_model_and_tokenizer
from utils.training import train_model, evaluate, measure_inference_latency
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

def train_lstm_cnn_model(data_dict, model_dir, caller_only=False, device=None, max_epochs=15):
    """Train LSTM-CNN model."""
    
    model_type = 'lstm-cnn'
    model_name = f"lstm_cnn_{'caller_only' if caller_only else 'full'}"
    print(f"Training {model_name} model...")
    
    # Prepare dataloaders
    dataloaders = prepare_dataloaders(
        data_dict, 
        model_type=model_type, 
        batch_size=32, 
        caller_only=caller_only
    )
    
    # Get vocabulary size
    vocab_size = len(dataloaders['vocab'])
    
    # Create model
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
    
    # Move model to device
    model = model.to(device)
    
    # Calculate class weights for handling imbalanced data
    class_weights = get_class_weights(data_dict['train']['labels'])
    class_weights = class_weights.to(device)
    
    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Create model save path
    model_save_path = os.path.join(model_dir, f"{model_name}.pt")
    
    # Train model with custom max_epochs
    model, history = train_model(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=max_epochs,  # Use the provided max_epochs
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
    
    # Measure inference latency
    latency_metrics = measure_inference_latency(model, dataloaders['test'], device)
    print(f"Inference latency for {model_name}:")
    print(f"Average: {latency_metrics['avg_latency'] * 1000:.2f} ms")
    print(f"95th percentile: {latency_metrics['p95_latency'] * 1000:.2f} ms")
    
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
    
    # Save vocabulary - CRITICAL FIX: Save vocabulary for inference
    vocab_save_path = os.path.join(model_dir, f"{model_name}_vocab.json")
    with open(vocab_save_path, 'w') as f:
        vocab_dict = {
            'word2idx': {k: v for k, v in dataloaders['vocab'].word2idx.items()},
            'idx2word': {str(k): v for k, v in dataloaders['vocab'].idx2word.items()} # Convert int keys to strings for JSON
        }
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary saved to {vocab_save_path}")
    
    # Save model metadata
    metadata = {
        'model_type': model_type,
        'caller_only': caller_only,
        'vocab_size': vocab_size,
        'embedding_dim': 300,
        'hidden_dim': 256,
        'num_layers': 2,
        'cnn_out_channels': 100,
        'kernel_sizes': [3, 5, 7],
        'dropout': 0.5,
        'training_history': {k: [float(val) for val in v] for k, v in history.items()},
        'test_metrics': {k: float(v) if not isinstance(v, type(test_metrics['confusion_matrix'])) else v.tolist() 
                         for k, v in test_metrics.items()},
        'latency_metrics': {k: float(v) if k != 'raw_latencies' else None for k, v in latency_metrics.items()},
    }
    
    # Save metadata
    metadata_save_path = os.path.join(model_dir, f"{model_name}_metadata.json")
    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return {
        'model': model,
        'test_metrics': test_metrics,
        'latency_metrics': latency_metrics,
        'vocab': dataloaders['vocab']
    }

def train_mlm_model(data_dict, model_dir, caller_only=False, device=None, max_epochs=10):
    """Train MLM-based model."""
    
    model_type = 'mlm'
    model_name = f"mlm_{'caller_only' if caller_only else 'full'}"
    print(f"Training {model_name} model...")
    
    # Get pre-trained model and tokenizer
    model, tokenizer = get_mlm_model_and_tokenizer()
    
    # Prepare dataloaders
    dataloaders = prepare_dataloaders(
        data_dict, 
        tokenizer=tokenizer,
        model_type=model_type, 
        batch_size=16,  # Smaller batch size due to memory constraints
        caller_only=caller_only
    )
    
    # Move model to device
    model = model.to(device)
    
    # Calculate class weights for handling imbalanced data
    class_weights = get_class_weights(data_dict['train']['labels'])
    class_weights = class_weights.to(device)
    
    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Create model save path
    model_save_path = os.path.join(model_dir, f"{model_name}.pt")
    
    # Train model with custom max_epochs
    model, history = train_model(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=max_epochs,  # Use the provided max_epochs
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
    
    # Measure inference latency
    latency_metrics = measure_inference_latency(model, dataloaders['test'], device)
    print(f"Inference latency for {model_name}:")
    print(f"Average: {latency_metrics['avg_latency'] * 1000:.2f} ms")
    print(f"95th percentile: {latency_metrics['p95_latency'] * 1000:.2f} ms")
    
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
        'model_type': model_type,
        'caller_only': caller_only,
        'pretrained_model': "jinaai/jina-embeddings-v2-small-en",
        'training_history': {k: [float(val) for val in v] for k, v in history.items()},
        'test_metrics': {k: float(v) if not isinstance(v, type(test_metrics['confusion_matrix'])) else v.tolist() 
                         for k, v in test_metrics.items()},
        'latency_metrics': {k: float(v) if k != 'raw_latencies' else None for k, v in latency_metrics.items()},
    }
    
    # Save metadata
    metadata_save_path = os.path.join(model_dir, f"{model_name}_metadata.json")
    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Save tokenizer
    tokenizer_save_path = os.path.join(model_dir, f"{model_name}_tokenizer")
    tokenizer.save_pretrained(tokenizer_save_path)
    
    return {
        'model': model,
        'test_metrics': test_metrics,
        'latency_metrics': latency_metrics,
        'tokenizer': tokenizer
    }

def train_and_evaluate_all_models(data_file, output_dir, models=None, max_rows=None, 
                                max_epochs=None, batch_size=32, train_size=0.7, 
                                val_size=0.15, save_vocab=True):
    """
    Train and evaluate all models specified.
    
    Args:
        data_file (str): Path to the data file.
        output_dir (str): Directory to save models and results.
        models (list): List of models to train. Options: 'lstm_cnn_full', 'lstm_cnn_caller_only', 'mlm_full', 'mlm_caller_only'.
        max_rows (int): Maximum number of rows to use from data file.
        max_epochs (int): Maximum number of epochs to train.
        batch_size (int): Batch size for training.
        train_size (float): Proportion of data to use for training.
        val_size (float): Proportion of data to use for validation.
        save_vocab (bool): Whether to save vocabulary for LSTM-CNN models.
        
    Returns:
        dict: Dictionary containing results for all trained models.
    """
    # Set max_epochs for each model
    lstm_cnn_epochs = 1 if max_epochs is None else max_epochs
    mlm_epochs = 1 if max_epochs is None else max_epochs
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    test_size = 1-train_size-val_size
    data_dict = load_and_preprocess_data(data_file, test_size=test_size, val_size=val_size)
    
    # If max_rows is specified, limit the data
    if max_rows is not None:
        print(f"Limiting to {max_rows} rows for testing")
        # Create a smaller subset for testing
        import random
        random.seed(42)
        
        train_indices = random.sample(range(len(data_dict['train']['texts'])), 
                                      min(max_rows, len(data_dict['train']['texts'])))
        val_indices = random.sample(range(len(data_dict['val']['texts'])), 
                                    min(max_rows // 5, len(data_dict['val']['texts'])))
        test_indices = random.sample(range(len(data_dict['test']['texts'])), 
                                     min(max_rows // 5, len(data_dict['test']['texts'])))
        
        data_dict['train']['texts'] = [data_dict['train']['texts'][i] for i in train_indices]
        data_dict['train']['labels'] = [data_dict['train']['labels'][i] for i in train_indices]
        data_dict['val']['texts'] = [data_dict['val']['texts'][i] for i in val_indices]
        data_dict['val']['labels'] = [data_dict['val']['labels'][i] for i in val_indices]
        data_dict['test']['texts'] = [data_dict['test']['texts'][i] for i in test_indices]
        data_dict['test']['labels'] = [data_dict['test']['labels'][i] for i in test_indices]
    
    # Print dataset statistics
    print("Dataset statistics:")
    print(f"Total samples: {len(data_dict['full_df'])}")
    print(f"Training samples: {len(data_dict['train']['texts'])}")
    print(f"Validation samples: {len(data_dict['val']['texts'])}")
    print(f"Test samples: {len(data_dict['test']['texts'])}")
    
    complaint_count = sum(data_dict['full_df']['binary_label'])
    print(f"Complaint ratio: {complaint_count / len(data_dict['full_df']) * 100:.2f}%")
    
    # Train and evaluate models
    models_results = {}
    
    # Define which models to train
    if models is None:
        models = ['lstm_cnn_full', 'lstm_cnn_caller_only', 'mlm_full', 'mlm_caller_only']
    
    # Make sure models is a list
    if isinstance(models, str):
        models = [models]
    
    # Train LSTM-CNN full conversation model
    if 'lstm_cnn_full' in models:
        start_time = datetime.now()
        lstm_cnn_full_results = train_lstm_cnn_model(data_dict, output_dir, caller_only=False, device=device, max_epochs=lstm_cnn_epochs)
        training_time = (datetime.now() - start_time).total_seconds()
        
        models_results['lstm_cnn_full'] = {
            'accuracy': lstm_cnn_full_results['test_metrics']['accuracy'],
            'precision': lstm_cnn_full_results['test_metrics']['precision'],
            'recall': lstm_cnn_full_results['test_metrics']['recall'],
            'f1': lstm_cnn_full_results['test_metrics']['f1'],
            'training_time': training_time,
            'inference_latency': lstm_cnn_full_results['latency_metrics']['avg_latency'] * 1000  # Convert to ms
        }
    
    # Train LSTM-CNN caller-only model
    if 'lstm_cnn_caller_only' in models:
        start_time = datetime.now()
        lstm_cnn_caller_results = train_lstm_cnn_model(data_dict, output_dir, caller_only=True, device=device, max_epochs=lstm_cnn_epochs)
        training_time = (datetime.now() - start_time).total_seconds()
        
        models_results['lstm_cnn_caller_only'] = {
            'accuracy': lstm_cnn_caller_results['test_metrics']['accuracy'],
            'precision': lstm_cnn_caller_results['test_metrics']['precision'],
            'recall': lstm_cnn_caller_results['test_metrics']['recall'],
            'f1': lstm_cnn_caller_results['test_metrics']['f1'],
            'training_time': training_time,
            'inference_latency': lstm_cnn_caller_results['latency_metrics']['avg_latency'] * 1000  # Convert to ms
        }
    
    # Train MLM full conversation model
    if 'mlm_full' in models:
        start_time = datetime.now()
        mlm_full_results = train_mlm_model(data_dict, output_dir, caller_only=False, device=device, max_epochs=mlm_epochs)
        training_time = (datetime.now() - start_time).total_seconds()
        
        models_results['mlm_full'] = {
            'accuracy': mlm_full_results['test_metrics']['accuracy'],
            'precision': mlm_full_results['test_metrics']['precision'],
            'recall': mlm_full_results['test_metrics']['recall'],
            'f1': mlm_full_results['test_metrics']['f1'],
            'training_time': training_time,
            'inference_latency': mlm_full_results['latency_metrics']['avg_latency'] * 1000  # Convert to ms
        }
    
    # Train MLM caller-only model
    if 'mlm_caller_only' in models:
        start_time = datetime.now()
        mlm_caller_results = train_mlm_model(data_dict, output_dir, caller_only=True, device=device, max_epochs=mlm_epochs)
        training_time = (datetime.now() - start_time).total_seconds()
        
        models_results['mlm_caller_only'] = {
            'accuracy': mlm_caller_results['test_metrics']['accuracy'],
            'precision': mlm_caller_results['test_metrics']['precision'],
            'recall': mlm_caller_results['test_metrics']['recall'],
            'f1': mlm_caller_results['test_metrics']['f1'],
            'training_time': training_time,
            'inference_latency': mlm_caller_results['latency_metrics']['avg_latency'] * 1000  # Convert to ms
        }
    
    # Compare models if multiple models were trained
    if len(models_results) > 1:
        # Convert to format expected by comparison functions
        test_metrics = {}
        latency_metrics = {}
        
        for model_name, metrics in models_results.items():
            display_name = model_name.replace('_', ' ').title()
            test_metrics[display_name] = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            }
            
            latency_metrics[display_name] = {
                'avg_latency': metrics['inference_latency'] / 1000  # Convert back to seconds
            }
        
        # Generate comparison table
        comparison_table = generate_model_comparison_table(test_metrics, latency_metrics)
        print("\nModel Comparison:")
        print(comparison_table)
        
        # Save comparison table
        comparison_table_path = os.path.join(output_dir, "model_comparison.csv")
        comparison_table.to_csv(comparison_table_path, index=False)
        
        # Plot F1 score comparison
        f1_comparison_path = os.path.join(output_dir, "f1_comparison.png")
        compare_models(test_metrics, metric_name='f1', save_path=f1_comparison_path)
        
        # Plot latency comparison
        latency_comparison_path = os.path.join(output_dir, "latency_comparison.png")
        plot_latency_comparison(latency_metrics, save_path=latency_comparison_path)
    
    print(f"All models and results saved to {output_dir}")
    return models_results

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate complaint detection models')
    parser.add_argument('--data_file', type=str, default='train.csv', help='Path to the data file')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for models and results')
    parser.add_argument('--models', type=str, default='all', 
                        choices=['all', 'lstm_cnn_full', 'lstm_cnn_caller_only', 'mlm_full', 'mlm_caller_only'],
                        help='Which models to train')
    parser.add_argument('--max_rows', type=int, default=None, help='Maximum number of rows to load (for testing)')
    parser.add_argument('--max_epochs', type=int, default=None, help='Maximum number of epochs to train (for testing)')
    parser.add_argument('--train_size', type=float, default=0.7, help='Proportion of data for training')
    parser.add_argument('--val_size', type=float, default=0.15, help='Proportion of data for validation')
    args = parser.parse_args()
    
    # Set max_epochs for each model
    lstm_cnn_epochs = 1 if args.max_epochs is not None else 15
    mlm_epochs = 1 if args.max_epochs is not None else 10
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(args.output_dir, f"models_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    
    # Use load_and_preprocess_data from data.preprocessor
    test_size = 1-args.train_size-args.val_size
    data_dict = load_and_preprocess_data(args.data_file, test_size=test_size, val_size=args.val_size)
    
    # If max_rows is specified, limit the data
    if args.max_rows is not None:
        print(f"Limiting to {args.max_rows} rows for testing")
        # Create a smaller subset for testing
        import random
        random.seed(42)
        
        train_indices = random.sample(range(len(data_dict['train']['texts'])), 
                                      min(args.max_rows, len(data_dict['train']['texts'])))
        val_indices = random.sample(range(len(data_dict['val']['texts'])), 
                                    min(args.max_rows // 5, len(data_dict['val']['texts'])))
        test_indices = random.sample(range(len(data_dict['test']['texts'])), 
                                     min(args.max_rows // 5, len(data_dict['test']['texts'])))
        
        data_dict['train']['texts'] = [data_dict['train']['texts'][i] for i in train_indices]
        data_dict['train']['labels'] = [data_dict['train']['labels'][i] for i in train_indices]
        data_dict['val']['texts'] = [data_dict['val']['texts'][i] for i in val_indices]
        data_dict['val']['labels'] = [data_dict['val']['labels'][i] for i in val_indices]
        data_dict['test']['texts'] = [data_dict['test']['texts'][i] for i in test_indices]
        data_dict['test']['labels'] = [data_dict['test']['labels'][i] for i in test_indices]
    
    # Print dataset statistics
    print("Dataset statistics:")
    print(f"Total samples: {len(data_dict['full_df'])}")
    print(f"Training samples: {len(data_dict['train']['texts'])}")
    print(f"Validation samples: {len(data_dict['val']['texts'])}")
    print(f"Test samples: {len(data_dict['test']['texts'])}")
    
    complaint_count = sum(data_dict['full_df']['binary_label'])
    print(f"Complaint ratio: {complaint_count / len(data_dict['full_df']) * 100:.2f}%")
    
    # Train and evaluate models
    models_results = {}
    
    # Train LSTM-CNN full conversation model
    if args.models in ['all', 'lstm_cnn_full']:
        lstm_cnn_full_results = train_lstm_cnn_model(data_dict, model_dir, caller_only=False, device=device, max_epochs=lstm_cnn_epochs)
        models_results['LSTM-CNN (Full)'] = lstm_cnn_full_results
    
    # Train LSTM-CNN caller-only model
    if args.models in ['all', 'lstm_cnn_caller_only']:
        lstm_cnn_caller_results = train_lstm_cnn_model(data_dict, model_dir, caller_only=True, device=device, max_epochs=lstm_cnn_epochs)
        models_results['LSTM-CNN (Caller-Only)'] = lstm_cnn_caller_results
    
    # Train MLM full conversation model
    if args.models in ['all', 'mlm_full']:
        mlm_full_results = train_mlm_model(data_dict, model_dir, caller_only=False, device=device, max_epochs=mlm_epochs)
        models_results['MLM (Full)'] = mlm_full_results
    
    # Train MLM caller-only model
    if args.models in ['all', 'mlm_caller_only']:
        mlm_caller_results = train_mlm_model(data_dict, model_dir, caller_only=True, device=device, max_epochs=mlm_epochs)
        models_results['MLM (Caller-Only)'] = mlm_caller_results
    
    # Compare models if multiple models were trained
    if len(models_results) > 1:
        # Extract metrics for comparison
        test_metrics = {name: results['test_metrics'] for name, results in models_results.items()}
        latency_metrics = {name: results['latency_metrics'] for name, results in models_results.items()}
        
        # Generate comparison table
        comparison_table = generate_model_comparison_table(test_metrics, latency_metrics)
        print("\nModel Comparison:")
        print(comparison_table)
        
        # Save comparison table
        comparison_table_path = os.path.join(model_dir, "model_comparison.csv")
        comparison_table.to_csv(comparison_table_path, index=False)
        
        # Plot F1 score comparison
        f1_comparison_path = os.path.join(model_dir, "f1_comparison.png")
        compare_models(test_metrics, metric_name='f1', save_path=f1_comparison_path)
        
        # Plot latency comparison
        latency_comparison_path = os.path.join(model_dir, "latency_comparison.png")
        plot_latency_comparison(latency_metrics, save_path=latency_comparison_path)
    
    print(f"All models and results saved to {model_dir}")

if __name__ == '__main__':
    main() 