"""
Model comparison script for training and evaluating all model types
on the synthetic call center data, and comparing their performance.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import time
import gc
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add parent directory to path for imports
sys.path.append('.')

# Import project modules
from utils.data_preprocessing import load_and_preprocess_data, prepare_data_pipeline
from models import get_model
from inference.inference import ComplaintDetector
from visualization.visualization import generate_dashboard, generate_report

import config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and compare models for complaint detection")
    parser.add_argument("--data_path", type=str, default="data/synthetic_call_center_data.csv",
                        help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default=os.path.join("models", "comparison", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                        help="Directory to save model outputs")
    parser.add_argument("--model_types", type=str, nargs="+", 
                        default=["mlm_full", "mlm_caller_only", "lstm_cnn_full", "lstm_cnn_caller_only", "hybrid_full", "hybrid_caller_only"],
                        help="Models to train and compare")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs to train each model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--generate_report", action="store_true",
                        help="Generate HTML report of comparison results")
    return parser.parse_args()

def train_model(model_type, train_dataloader, val_dataloader, device, output_dir, num_epochs):
    """Train a model and return its performance history."""
    # Create model
    model = get_model(model_type).to(device)
    
    # Setup optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Calculate class weights for imbalanced data
    # Use the weights from config, or calculate dynamically if needed
    pos_weight = torch.tensor([config.CLASS_WEIGHTS[1] / config.CLASS_WEIGHTS[0]]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Model path
    model_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_type}.pt")
    
    # Training history
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_f1": []
    }
    
    # Early stopping variables
    best_val_f1 = 0
    patience_counter = 0
    
    # Train model
    print(f"Training {model_type} model...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        train_loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)")
        for batch in train_loop:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].float().to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            
            # Get the logit for the positive class (index 1)
            outputs = outputs[:, 1]  # Get the logit for the positive class
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).int()
            correct += (predictions == labels.int()).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            train_loop.set_postfix(loss=train_loss/len(train_loop), accuracy=correct/total)
        
        # Calculate training metrics
        train_loss /= len(train_dataloader)
        train_accuracy = correct / total
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_predictions = []
        val_labels = []
        
        val_loop = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)")
        with torch.no_grad():
            for batch in val_loop:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].float().to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                
                # Get the logit for the positive class (index 1)
                outputs = outputs[:, 1]  # Get the logit for the positive class
                loss = criterion(outputs, labels)
                
                # Track metrics
                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).int()
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.int().cpu().numpy())
                
                # Update progress bar
                val_loop.set_postfix(loss=val_loss/len(val_loop))
        
        # Calculate validation metrics
        val_loss /= len(val_dataloader)
        val_f1 = f1_score(val_labels, val_predictions, zero_division=0)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        # Save checkpoint if this is the best model so far
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"  Model saved to {model_path} (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Save training history
    history_path = os.path.join(model_dir, f"{model_type}_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)
    
    # Always save the final model state regardless of performance
    # This ensures we have a model file even if validation F1 never improved
    if not os.path.exists(model_path):
        torch.save(model.state_dict(), model_path)
        print(f"  Final model saved to {model_path}")
    
    return model_path, history

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model_path):
    """Get the size of a model file in MB."""
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def measure_inference_latency(model, tokenizer, device, num_samples=100, max_length=512):
    """
    Measure the average inference latency of a model.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer for encoding text
        device: Device to run inference on
        num_samples: Number of samples to test
        max_length: Maximum sequence length
        
    Returns:
        dict: Latency metrics in milliseconds
    """
    model.eval()
    
    # Generate dummy input of realistic length
    text = "This is a sample text for measuring inference latency. " * 10
    
    # Process input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length")
    
    # Remove token_type_ids if present as MLMComplaintDetector doesn't use it
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    
    # Move inputs to device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_ids, attention_mask)
    
    # Measure latency
    latencies = []
    with torch.no_grad():
        for _ in range(num_samples):
            start_time = time.time()
            _ = model(input_ids, attention_mask)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency)
    
    # Calculate statistics
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    
    return {
        "avg_latency_ms": avg_latency,
        "std_latency_ms": std_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency
    }

def evaluate_model(model_type, model_path, test_dataloader, device):
    """Evaluate a trained model on the test set."""
    # Load model
    model = get_model(model_type).to(device)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        # Return empty metrics
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": np.zeros((2, 2)),
            "model_size_mb": 0.0,
            "param_count": 0,
            "latency_metrics": {
                "avg_latency_ms": 0.0,
                "std_latency_ms": 0.0,
                "min_latency_ms": 0.0,
                "max_latency_ms": 0.0
            }
        }
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Count parameters and get model size
    param_count = count_parameters(model)
    model_size_mb = get_model_size_mb(model_path)
    
    # Get tokenizer for latency measurement
    tokenizer = get_tokenizer(model_type)
    
    # Measure inference latency
    latency_metrics = measure_inference_latency(model, tokenizer, device)
    
    # Evaluation
    test_predictions = []
    test_labels = []
    test_probs = []
    
    # Set a lower threshold for imbalanced data
    threshold = 0.3  # Lower threshold to increase recall
    
    test_loop = tqdm(test_dataloader, desc=f"Evaluating {model_type}")
    with torch.no_grad():
        for batch in test_loop:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].float().to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Get the logit for the positive class (index 1)
            outputs = outputs[:, 1]  # Get the logit for the positive class
            
            # Get predictions
            probs = torch.sigmoid(outputs)
            predictions = (probs > threshold).int()  # Use lower threshold
            
            # Store results
            test_predictions.extend(predictions.cpu().numpy())
            test_labels.extend(labels.int().cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions, zero_division=0)
    recall = recall_score(test_labels, test_predictions, zero_division=0)
    f1 = f1_score(test_labels, test_predictions, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_predictions)
    
    # Print detailed performance report
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Classification threshold: {threshold}")
    
    if cm.size > 1:  # Only print if we have a proper 2x2 matrix
        tn, fp, fn, tp = cm.ravel()
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # Print model size and latency information
    print(f"  Model Size: {model_size_mb:.2f} MB")
    print(f"  Parameter Count: {param_count:,}")
    print(f"  Avg. Inference Latency: {latency_metrics['avg_latency_ms']:.2f} ms")
    
    # Return metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "threshold": threshold,
        "probabilities": test_probs,
        "model_size_mb": model_size_mb,
        "param_count": param_count,
        "latency_metrics": latency_metrics
    }
    
    return metrics

def predict_with_model(model_type, model_path, df, tokenizer, device):
    """Make predictions with a trained model."""
    # Load model
    model = get_model(model_type).to(device)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        # Return dataframe with empty predictions
        df["prediction"] = 0
        df["probability"] = 0.0
        return df
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Use the same threshold as in evaluation
    threshold = 0.3
    
    # Create detector
    detector = ComplaintDetector(model_type=model_type, model_path=model_path)
    
    # Make predictions
    predictions = []
    probabilities = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Predicting with {model_type}"):
        text = row["text"] if "text" in df.columns else row["full_text"]
        
        # Make prediction
        is_complaint, prob, _ = detector.predict(text)
        
        # Apply the lower threshold to raw probability
        is_complaint_adjusted = prob >= threshold
        
        # Store results
        predictions.append(int(is_complaint_adjusted))
        probabilities.append(float(prob))
    
    # Add predictions to dataframe
    df_copy = df.copy()
    df_copy["is_complaint"] = predictions
    df_copy["complaint_probability"] = probabilities
    df_copy["threshold_used"] = threshold
    
    # Add intensity labels based on probability
    def get_intensity(prob):
        if prob < 0.25:
            return "Low"
        elif prob < 0.5:
            return "Medium"
        elif prob < 0.75:
            return "High"
        else:
            return "Very High"
    
    df_copy["complaint_intensity"] = df_copy["complaint_probability"].apply(get_intensity)
    
    return df_copy

def plot_training_histories(histories, output_path):
    """Plot training and validation metrics for multiple models."""
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot training loss
    ax = axes[0, 0]
    for model_type, history in histories.items():
        ax.plot(history["train_loss"], label=model_type)
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot validation loss
    ax = axes[0, 1]
    for model_type, history in histories.items():
        ax.plot(history["val_loss"], label=model_type)
    ax.set_title("Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot training accuracy
    ax = axes[1, 0]
    for model_type, history in histories.items():
        ax.plot(history["train_accuracy"], label=model_type)
    ax.set_title("Training Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot validation F1
    ax = axes[1, 1]
    for model_type, history in histories.items():
        ax.plot(history["val_f1"], label=model_type)
    ax.set_title("Validation F1 Score")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return fig

def plot_comparison_bar(metrics_dict, output_path):
    """Plot bar chart comparing model metrics."""
    # Extract metrics
    model_types = list(metrics_dict.keys())
    f1_scores = [metrics_dict[model_type]["f1"] for model_type in model_types]
    precision_scores = [metrics_dict[model_type]["precision"] for model_type in model_types]
    recall_scores = [metrics_dict[model_type]["recall"] for model_type in model_types]
    
    # Shorten model names for plotting
    short_names = [model_type.replace("_", "\n") for model_type in model_types]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot metrics
    x = np.arange(len(model_types))
    width = 0.25
    
    plt.bar(x - width, f1_scores, width, label="F1 Score", color="blue")
    plt.bar(x, precision_scores, width, label="Precision", color="green")
    plt.bar(x + width, recall_scores, width, label="Recall", color="orange")
    
    # Add labels and legend
    plt.xlabel("Model Type")
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.xticks(x, short_names)
    plt.ylim(0, 1.1)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # Create model size and latency comparison
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot model sizes
    model_sizes = [metrics_dict[model_type]["model_size_mb"] for model_type in model_types]
    ax1.bar(short_names, model_sizes, color="purple")
    ax1.set_title("Model Size Comparison")
    ax1.set_ylabel("Size (MB)")
    ax1.set_xlabel("Model Type")
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot inference latencies
    latencies = [metrics_dict[model_type]["latency_metrics"]["avg_latency_ms"] for model_type in model_types]
    ax2.bar(short_names, latencies, color="crimson")
    ax2.set_title("Inference Latency Comparison")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_xlabel("Model Type")
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    size_latency_path = os.path.splitext(output_path)[0] + "_size_latency.png"
    plt.savefig(size_latency_path, dpi=300)
    plt.close()

def plot_confusion_matrices(metrics_dict, output_path):
    """Plot confusion matrices for all models."""
    # Determine number of rows and columns for subplot grid
    n_models = len(metrics_dict)
    n_cols = 3  # Number of columns in the grid
    n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    # Flatten axes if there's only one row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot confusion matrices
    for i, (model, metrics) in enumerate(metrics_dict.items()):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        cm = metrics["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                   xticklabels=["Non-Complaint", "Complaint"],
                   yticklabels=["Non-Complaint", "Complaint"])
        ax.set_title(f"{model}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    
    # Hide unused subplots
    for i in range(n_models, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return fig

def generate_comparison_report(metrics_dict, histories, prediction_dfs, output_dir):
    """Generate HTML report of model comparison results."""
    # Create report directory
    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate comparison charts
    metrics_path = os.path.join(report_dir, "metrics_comparison.png")
    plot_comparison_bar(metrics_dict, metrics_path)
    
    cm_path = os.path.join(report_dir, "confusion_matrices.png")
    plot_confusion_matrices(metrics_dict, cm_path)
    
    histories_path = os.path.join(report_dir, "training_histories.png")
    plot_training_histories(histories, histories_path)
    
    # Generate HTML report
    report_path = os.path.join(report_dir, "model_comparison_report.html")
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .chart {{
                max-width: 100%;
                margin: 20px 0;
            }}
            .highlight {{
                font-weight: bold;
                color: #2e8b57;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Comparison Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Performance Metrics</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>F1 Score</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Accuracy</th>
                    <th>Model Size (MB)</th>
                    <th>Parameters</th>
                    <th>Inference Latency (ms)</th>
                </tr>
    """
    
    # Find best model by F1 score
    best_model = max(metrics_dict.keys(), key=lambda model: metrics_dict[model]["f1"])
    
    # Add rows for each model
    for model_type, metrics in metrics_dict.items():
        is_best = model_type == best_model
        f1 = metrics["f1"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        accuracy = metrics["accuracy"]
        model_size = metrics["model_size_mb"]
        param_count = metrics["param_count"]
        latency = metrics["latency_metrics"]["avg_latency_ms"]
        
        row_class = "highlight" if is_best else ""
        html_content += f"""
                <tr class="{row_class}">
                    <td>{model_type}</td>
                    <td>{f1:.4f}</td>
                    <td>{precision:.4f}</td>
                    <td>{recall:.4f}</td>
                    <td>{accuracy:.4f}</td>
                    <td>{model_size:.2f}</td>
                    <td>{param_count:,}</td>
                    <td>{latency:.2f}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>Performance Comparison Charts</h2>
            <div class="chart">
                <img src="metrics_comparison.png" alt="Metrics Comparison" style="width: 100%;">
            </div>
            
            <h2>Model Size and Latency Comparison</h2>
            <div class="chart">
                <img src="metrics_comparison_size_latency.png" alt="Size and Latency Comparison" style="width: 100%;">
            </div>
            
            <h2>Confusion Matrices</h2>
            <div class="chart">
                <img src="confusion_matrices.png" alt="Confusion Matrices" style="width: 100%;">
            </div>
            
            <h2>Training Histories</h2>
            <div class="chart">
                <img src="training_histories.png" alt="Training Histories" style="width: 100%;">
            </div>
            
            <h2>Best Model</h2>
    """
    
    # Add best model details
    best_metrics = metrics_dict[best_model]
    html_content += f"""
            <p>The best performing model is <strong>{best_model}</strong> with an F1 score of {best_metrics["f1"]:.4f}.</p>
            <h3>Model Details</h3>
            <ul>
                <li><strong>F1 Score:</strong> {best_metrics["f1"]:.4f}</li>
                <li><strong>Precision:</strong> {best_metrics["precision"]:.4f}</li>
                <li><strong>Recall:</strong> {best_metrics["recall"]:.4f}</li>
                <li><strong>Accuracy:</strong> {best_metrics["accuracy"]:.4f}</li>
                <li><strong>Model Size:</strong> {best_metrics["model_size_mb"]:.2f} MB</li>
                <li><strong>Parameter Count:</strong> {best_metrics["param_count"]:,}</li>
                <li><strong>Average Inference Latency:</strong> {best_metrics["latency_metrics"]["avg_latency_ms"]:.2f} ms</li>
                <li><strong>Latency Standard Deviation:</strong> {best_metrics["latency_metrics"]["std_latency_ms"]:.2f} ms</li>
                <li><strong>Min Latency:</strong> {best_metrics["latency_metrics"]["min_latency_ms"]:.2f} ms</li>
                <li><strong>Max Latency:</strong> {best_metrics["latency_metrics"]["max_latency_ms"]:.2f} ms</li>
            </ul>
            
            <h3>Model Selection Considerations</h3>
            <p>
                When selecting a model for deployment, consider the trade-offs between accuracy metrics, 
                model size, and inference latency. For applications with strict latency requirements, 
                a smaller model with slightly lower performance might be preferable.
            </p>
            
            <h2>Detailed Model Analysis</h2>
    """
    
    # Add section for each model
    for model_type, metrics in metrics_dict.items():
        # Calculate TN, FP, FN, TP from confusion matrix
        if metrics["confusion_matrix"].size > 1:  # Only if we have a proper 2x2 matrix
            tn, fp, fn, tp = metrics["confusion_matrix"].ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        
        html_content += f"""
            <h3>{model_type}</h3>
            <h4>Performance Metrics</h4>
            <ul>
                <li><strong>F1 Score:</strong> {metrics["f1"]:.4f}</li>
                <li><strong>Precision:</strong> {metrics["precision"]:.4f}</li>
                <li><strong>Recall:</strong> {metrics["recall"]:.4f}</li>
                <li><strong>Accuracy:</strong> {metrics["accuracy"]:.4f}</li>
            </ul>
            
            <h4>Model Efficiency</h4>
            <ul>
                <li><strong>Model Size:</strong> {metrics["model_size_mb"]:.2f} MB</li>
                <li><strong>Parameter Count:</strong> {metrics["param_count"]:,}</li>
                <li><strong>Average Inference Latency:</strong> {metrics["latency_metrics"]["avg_latency_ms"]:.2f} ms</li>
            </ul>
            
            <h4>Confusion Matrix</h4>
            <table>
                <tr>
                    <th></th>
                    <th>Predicted Negative</th>
                    <th>Predicted Positive</th>
                </tr>
                <tr>
                    <th>Actual Negative</th>
                    <td>TN: {tn}</td>
                    <td>FP: {fp}</td>
                </tr>
                <tr>
                    <th>Actual Positive</th>
                    <td>FN: {fn}</td>
                    <td>TP: {tp}</td>
                </tr>
            </table>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(report_path, "w") as f:
        f.write(html_content)
    
    print(f"Model comparison report generated at {report_path}")

def main():
    """Main function for model training and comparison."""
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(args.data_path)
    
    # Prepare dataloaders for different model types
    dataloaders = {}
    tokenizers = {}
    for model_type in args.model_types:
        print(f"Preparing data for {model_type}...")
        train_dataloader, val_dataloader, test_dataloader, tokenizer = prepare_data_pipeline(
            df, 
            model_type=model_type,
            batch_size=args.batch_size
        )
        dataloaders[model_type] = (train_dataloader, val_dataloader, test_dataloader)
        tokenizers[model_type] = tokenizer
        
    # Train and evaluate models
    model_paths = {}
    histories = {}
    metrics_dict = {}
    prediction_dfs = {}
    
    for model_type in args.model_types:
        print(f"\n{'='*50}")
        print(f"Processing {model_type} model...")
        print(f"{'='*50}")
        
        # Get dataloaders
        train_dataloader, val_dataloader, test_dataloader = dataloaders[model_type]
        
        # Train model
        model_path, history = train_model(
            model_type, 
            train_dataloader, 
            val_dataloader, 
            device, 
            args.output_dir,
            args.num_epochs
        )
        
        # Store model path and history
        model_paths[model_type] = model_path
        histories[model_type] = history
        
        # Evaluate model
        print("\nEvaluating model...")
        metrics = evaluate_model(model_type, model_path, test_dataloader, device)
        metrics_dict[model_type] = metrics
        
        # Make predictions on test set
        print("\nMaking predictions on test set...")
        test_df = predict_with_model(model_type, model_path, df, tokenizers[model_type], device)
        prediction_dfs[model_type] = test_df
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
    
    # Find best model by F1 score
    best_model = max(metrics_dict.keys(), key=lambda model: metrics_dict[model]["f1"])
    print(f"\nBest model by F1 score: {best_model} (F1: {metrics_dict[best_model]['f1']:.4f})")
    
    # Print model size and latency comparison
    print("\nModel Size and Latency Comparison:")
    print(f"{'Model Type':<20} {'Size (MB)':<12} {'Parameters':<15} {'Avg Latency (ms)':<16}")
    print("-" * 65)
    for model_type in args.model_types:
        metrics = metrics_dict[model_type]
        print(f"{model_type:<20} {metrics['model_size_mb']:<12.2f} {metrics['param_count']:<15,} {metrics['latency_metrics']['avg_latency_ms']:<16.2f}")
    
    # Generate comprehensive report
    if args.generate_report:
        print("\nGenerating model comparison report...")
        generate_comparison_report(metrics_dict, histories, prediction_dfs, args.output_dir)

def get_tokenizer(model_type):
    """Get the appropriate tokenizer for a model type."""
    from transformers import AutoTokenizer
    import config
    
    if model_type.startswith("mlm_") or model_type.startswith("hybrid_"):
        return AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
    else:
        # For LSTM-CNN models, use a basic tokenizer
        return AutoTokenizer.from_pretrained("bert-base-uncased")

if __name__ == "__main__":
    main() 