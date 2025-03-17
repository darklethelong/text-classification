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
    criterion = torch.nn.BCEWithLogitsLoss()
    
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
            "confusion_matrix": np.zeros((2, 2))
        }
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Evaluation
    test_predictions = []
    test_labels = []
    
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
            predictions = (probs > 0.5).int()
            
            # Store results
            test_predictions.extend(predictions.cpu().numpy())
            test_labels.extend(labels.int().cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions, zero_division=0)
    recall = recall_score(test_labels, test_predictions, zero_division=0)
    f1 = f1_score(test_labels, test_predictions, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_predictions)
    
    # Return metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
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
    
    # Create detector
    detector = ComplaintDetector(model_type=model_type, model_path=model_path)
    
    # Make predictions
    predictions = []
    probabilities = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Predicting with {model_type}"):
        text = row["text"] if "text" in df.columns else row["full_text"]
        
        # Make prediction
        is_complaint, prob, _ = detector.predict(text)
        
        # Store results
        predictions.append(int(is_complaint))
        probabilities.append(float(prob))
    
    # Add predictions to dataframe
    df_copy = df.copy()
    df_copy["is_complaint"] = predictions
    df_copy["complaint_probability"] = probabilities
    
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
    """Plot bar chart comparing model performance metrics."""
    # Create dataframe for plotting
    models = list(metrics_dict.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    df_data = []
    
    for model in models:
        for metric in metrics:
            df_data.append({
                "Model": model,
                "Metric": metric.capitalize(),
                "Value": metrics_dict[model][metric]
            })
    
    df_plot = pd.DataFrame(df_data)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Model", y="Value", hue="Metric", data=df_plot)
    plt.title("Model Performance Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return plt.gcf()

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
    """Generate HTML report comparing model performance."""
    # Create output directory for report
    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate plots
    history_plot_path = os.path.join(report_dir, "training_histories.png")
    plot_training_histories(histories, history_plot_path)
    
    comparison_plot_path = os.path.join(report_dir, "model_comparison.png")
    plot_comparison_bar(metrics_dict, comparison_plot_path)
    
    cm_plot_path = os.path.join(report_dir, "confusion_matrices.png")
    plot_confusion_matrices(metrics_dict, cm_plot_path)
    
    # Find best model
    best_model = max(metrics_dict.items(), key=lambda x: x[1]["f1"])
    best_model_name = best_model[0]
    best_model_f1 = best_model[1]["f1"]
    
    # Generate predictions visualizations for best model
    best_model_vis_dir = os.path.join(report_dir, "best_model_vis")
    os.makedirs(best_model_vis_dir, exist_ok=True)
    
    generate_dashboard(
        prediction_dfs[best_model_name],
        best_model_vis_dir,
        timestamp_column=None,
        text_column="text"
    )
    
    # Generate HTML report
    html_path = os.path.join(output_dir, "model_comparison_report.html")
    
    # Create model metrics table HTML
    metrics_table_html = """
    <table>
        <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1 Score</th>
        </tr>
    """
    
    for model, metrics in metrics_dict.items():
        # Highlight best model
        row_style = ' style="background-color: #e6ffe6;"' if model == best_model_name else ""
        
        metrics_table_html += f"""
        <tr{row_style}>
            <td>{model}</td>
            <td>{metrics['accuracy']:.4f}</td>
            <td>{metrics['precision']:.4f}</td>
            <td>{metrics['recall']:.4f}</td>
            <td>{metrics['f1']:.4f}</td>
        </tr>
        """
    
    metrics_table_html += "</table>"
    
    # Write HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #1a468e; color: white; padding: 20px; text-align: center; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
            img {{ max-width: 100%; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .highlight {{ color: #d9534f; font-weight: bold; }}
            .best-model {{ background-color: #dff0d8; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Model Comparison Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        
        <div class="section">
            <h2>Summary</h2>
            <p>
                This report compares the performance of {len(metrics_dict)} different model architectures for complaint detection.
                The best performing model was <span class="highlight">{best_model_name}</span> with an F1 score of <span class="highlight">{best_model_f1:.4f}</span>.
            </p>
        </div>
        
        <div class="section">
            <h2>Model Performance Metrics</h2>
            {metrics_table_html}
        </div>
        
        <div class="section">
            <h2>Performance Comparison</h2>
            <img src="report/model_comparison.png" alt="Model Performance Comparison">
        </div>
        
        <div class="section">
            <h2>Confusion Matrices</h2>
            <img src="report/confusion_matrices.png" alt="Confusion Matrices">
        </div>
        
        <div class="section">
            <h2>Training History</h2>
            <img src="report/training_histories.png" alt="Training Histories">
        </div>
        
        <div class="section">
            <h2>Best Model Visualizations</h2>
            <p>The following visualizations show the performance of the best model ({best_model_name}) on the test set:</p>
            <div>
                <img src="report/best_model_vis/complaint_gauge.png" alt="Complaint Percentage Gauge">
                <img src="report/best_model_vis/complaint_distribution.png" alt="Complaint Distribution">
            </div>
            <div>
                <img src="report/best_model_vis/complaint_timeline.png" alt="Complaint Timeline">
                <img src="report/best_model_vis/complaint_heatmap.png" alt="Complaint Heatmap">
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Comparison report generated at {html_path}")
    
    return html_path

def main():
    """Main function to run training and comparison."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Save all model types for consistent handling
    all_model_types = ["mlm_full", "mlm_caller_only", "lstm_cnn_full", 
                      "lstm_cnn_caller_only", "hybrid_full", "hybrid_caller_only"]
    
    # Filter to requested model types
    model_types = [m for m in all_model_types if m in args.model_types]
    print(f"Training and comparing the following models: {model_types}")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv(args.data_path)
    
    # Check if label column has text values (complaint/non-complaint) instead of 0/1
    if df["label"].dtype == object:
        # Convert to 0/1
        df["label"] = df["label"].map({"non-complaint": 0, "complaint": 1})
    
    # Add "text" column if it doesn't exist (using full_text)
    if "text" not in df.columns and "full_text" in df.columns:
        df["text"] = df["full_text"]
    
    # Preprocess the data directly using the load_and_preprocess_data function
    # Set the required columns in config temporarily
    original_text_column = config.TEXT_COLUMN
    config.TEXT_COLUMN = "text"
    
    # Process the data
    processed_df = load_and_preprocess_data(data_path=None, create_caller_only=True, df=df)
    
    # Restore the original config
    config.TEXT_COLUMN = original_text_column
    
    # Use the processed dataframe
    df = processed_df
    
    # Save a copy of processed data
    processed_data_path = os.path.join(args.output_dir, "processed_data.csv")
    df.to_csv(processed_data_path, index=False)
    
    # Prepare data for training
    # We'll need both full and caller-only versions
    train_dataloaders = {}
    val_dataloaders = {}
    test_dataloaders = {}
    tokenizers = {}
    
    # Prepare data for full text models
    train_dataloaders["full"], val_dataloaders["full"], test_dataloaders["full"], tokenizers["full"] = \
        prepare_data_pipeline(use_caller_only=False, data_df=df, batch_size=args.batch_size)
    
    # Prepare data for caller-only models
    train_dataloaders["caller_only"], val_dataloaders["caller_only"], test_dataloaders["caller_only"], tokenizers["caller_only"] = \
        prepare_data_pipeline(use_caller_only=True, data_df=df, batch_size=args.batch_size)
    
    # Train all models
    model_paths = {}
    histories = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training {model_type} model")
        print(f"{'='*50}")
        
        # Select appropriate dataloader based on model type
        dataloader_key = "caller_only" if "caller_only" in model_type else "full"
        
        # Train model
        model_path, history = train_model(
            model_type=model_type,
            train_dataloader=train_dataloaders[dataloader_key],
            val_dataloader=val_dataloaders[dataloader_key],
            device=device,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs
        )
        
        model_paths[model_type] = model_path
        histories[model_type] = history
    
    # Evaluate all models
    print("\nEvaluating models on test set...")
    metrics_dict = {}
    
    for model_type in model_types:
        print(f"Evaluating {model_type}...")
        
        # Select appropriate dataloader based on model type
        dataloader_key = "caller_only" if "caller_only" in model_type else "full"
        
        # Evaluate model
        metrics = evaluate_model(
            model_type=model_type,
            model_path=model_paths[model_type],
            test_dataloader=test_dataloaders[dataloader_key],
            device=device
        )
        
        metrics_dict[model_type] = metrics
        
        # Print metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
    
    # Make predictions with all models
    print("\nMaking predictions with all models...")
    prediction_dfs = {}
    
    for model_type in model_types:
        # Select appropriate tokenizer based on model type
        tokenizer_key = "caller_only" if "caller_only" in model_type else "full"
        
        # Make predictions
        predictions_df = predict_with_model(
            model_type=model_type,
            model_path=model_paths[model_type],
            df=df,
            tokenizer=tokenizers[tokenizer_key],
            device=device
        )
        
        # Save predictions
        predictions_path = os.path.join(args.output_dir, f"{model_type}_predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        
        prediction_dfs[model_type] = predictions_df
    
    # Compare models
    print("\nGenerating comparison visualizations...")
    
    # Plot training histories
    history_plot_path = os.path.join(args.output_dir, "training_histories.png")
    plot_training_histories(histories, history_plot_path)
    
    # Plot model comparison
    comparison_plot_path = os.path.join(args.output_dir, "model_comparison.png")
    plot_comparison_bar(metrics_dict, comparison_plot_path)
    
    # Plot confusion matrices
    cm_plot_path = os.path.join(args.output_dir, "confusion_matrices.png")
    plot_confusion_matrices(metrics_dict, cm_plot_path)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "model_metrics.csv")
    metrics_df = pd.DataFrame({
        "model_type": model_types,
        "accuracy": [metrics_dict[m]["accuracy"] for m in model_types],
        "precision": [metrics_dict[m]["precision"] for m in model_types],
        "recall": [metrics_dict[m]["recall"] for m in model_types],
        "f1": [metrics_dict[m]["f1"] for m in model_types]
    })
    metrics_df.to_csv(metrics_path, index=False)
    
    # Find best model
    best_model = max(metrics_dict.items(), key=lambda x: x[1]["f1"])
    best_model_name = best_model[0]
    best_model_f1 = best_model[1]["f1"]
    
    print(f"\nBest model: {best_model_name} with F1 score of {best_model_f1:.4f}")
    
    # Generate comparison report if requested
    if args.generate_report:
        report_path = generate_comparison_report(
            metrics_dict=metrics_dict,
            histories=histories,
            prediction_dfs=prediction_dfs,
            output_dir=args.output_dir
        )
        print(f"Report saved to {report_path}")
    
    print("\nComparison complete!")
    

if __name__ == "__main__":
    main() 