import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import pandas as pd

def plot_training_history(history, save_path=None):
    """Plot training history."""
    plt.figure(figsize=(12, 4))
    
    # Get the number of epochs
    epochs = len(history['train_loss'])
    epoch_range = list(range(1, epochs + 1))  # Generate proper x-axis values
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    
    # Handle single-epoch case properly
    if epochs == 1:
        # For single-epoch, use scatter plots instead of lines
        plt.scatter(epoch_range, history['train_loss'], label='Train', color='blue')
        plt.scatter(epoch_range, history['val_loss'], label='Validation', color='orange')
    else:
        plt.plot(epoch_range, history['train_loss'], label='Train')
        plt.plot(epoch_range, history['val_loss'], label='Validation')
    
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Ensure proper x-axis ticks for single epoch
    if epochs == 1:
        plt.xticks([1])
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    
    # Handle single-epoch case properly
    if epochs == 1:
        # For single-epoch, use scatter plot
        plt.scatter(epoch_range, history['val_acc'], color='blue')
    else:
        plt.plot(epoch_range, history['val_acc'])
    
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    # Ensure proper x-axis ticks for single epoch
    if epochs == 1:
        plt.xticks([1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, classes=['Non-Complaint', 'Complaint'], save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        save_path: Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_roc_curve(y_true, y_proba, save_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for the positive class
        save_path: Path to save the figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_precision_recall_curve(y_true, y_proba, save_path=None):
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for the positive class
        save_path: Path to save the figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def compare_models(models_metrics, metric_name='f1', save_path=None):
    """
    Compare multiple models based on a specific metric.
    
    Args:
        models_metrics: Dictionary with model names as keys and metric dictionaries as values
        metric_name: Name of the metric to compare
        save_path: Path to save the figure
    """
    model_names = list(models_metrics.keys())
    metric_values = [metrics[metric_name] for metrics in models_metrics.values()]
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette('viridis', len(model_names))
    
    bars = plt.bar(model_names, metric_values, color=colors)
    
    plt.xlabel('Models')
    plt.ylabel(f'{metric_name.capitalize()} Score')
    plt.title(f'Model Comparison by {metric_name.capitalize()}')
    plt.ylim(0, 1.0)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_latency_comparison(cpu_latencies, gpu_latencies=None, save_path=None):
    """Plot latency comparison between models for both CPU and GPU."""
    plt.figure(figsize=(10, 6))
    
    # Set up the models and colors
    models = list(cpu_latencies.keys())
    n_models = len(models)
    bar_width = 0.35
    
    # Position of bars on x-axis
    r1 = np.arange(n_models)
    r2 = [x + bar_width for x in r1] if gpu_latencies else None
    
    # CPU latencies (in milliseconds)
    cpu_avg_latencies = [cpu_latencies[model]['avg_latency'] * 1000 for model in models]
    
    # Plot CPU latencies
    cpu_bars = plt.bar(r1, cpu_avg_latencies, width=bar_width, label='CPU', color='blue', alpha=0.7)
    
    # Plot GPU latencies if available
    if gpu_latencies:
        # Some models might not have GPU metrics if they failed or weren't measured
        gpu_avg_latencies = []
        for model in models:
            if model in gpu_latencies and gpu_latencies[model] is not None:
                gpu_avg_latencies.append(gpu_latencies[model]['avg_latency'] * 1000)
            else:
                gpu_avg_latencies.append(0)  # Use 0 as a placeholder for models without GPU metrics
        
        gpu_bars = plt.bar(r2, gpu_avg_latencies, width=bar_width, label='GPU', color='orange', alpha=0.7)
    
    # Add labels, title and axes ticks
    plt.xlabel('Model')
    plt.ylabel('Average Latency (ms)')
    plt.title('Inference Latency Comparison')
    plt.xticks([r + bar_width/2 for r in range(n_models)] if gpu_latencies else r1, models)
    plt.legend()
    
    # Add latency values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:  # Only add label if there's a value
                plt.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    autolabel(cpu_bars)
    if gpu_latencies:
        autolabel(gpu_bars)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def generate_model_comparison_table(models_metrics, cpu_latencies=None, gpu_latencies=None):
    """Generate a table comparing model metrics."""
    # Create DataFrame
    model_names = list(models_metrics.keys())
    
    # Create DataFrame with metrics
    df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': [models_metrics[model]['accuracy'] for model in model_names],
        'Precision': [models_metrics[model]['precision'] for model in model_names],
        'Recall': [models_metrics[model]['recall'] for model in model_names],
        'F1 Score': [models_metrics[model]['f1'] for model in model_names]
    })
    
    # Add CPU latency if provided
    if cpu_latencies:
        df['CPU Avg Latency (ms)'] = [cpu_latencies[model]['avg_latency'] * 1000 for model in model_names]
        df['CPU P95 Latency (ms)'] = [cpu_latencies[model]['p95_latency'] * 1000 for model in model_names]
    
    # Add GPU latency if provided
    if gpu_latencies:
        # Handle cases where some models might not have GPU metrics
        gpu_avg_latencies = []
        gpu_p95_latencies = []
        
        for model in model_names:
            if model in gpu_latencies and gpu_latencies[model] is not None:
                gpu_avg_latencies.append(gpu_latencies[model]['avg_latency'] * 1000)
                gpu_p95_latencies.append(gpu_latencies[model]['p95_latency'] * 1000)
            else:
                gpu_avg_latencies.append(None)  # Use None for missing values
                gpu_p95_latencies.append(None)
        
        df['GPU Avg Latency (ms)'] = gpu_avg_latencies
        df['GPU P95 Latency (ms)'] = gpu_p95_latencies
    
    return df 