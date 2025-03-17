"""
Evaluation module for complaint detection models.
Provides functions for evaluating models and comparing performance.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import os
import time
from tqdm import tqdm

import sys
sys.path.append('.')
import config
from utils.training import evaluate


def predict(model, dataloader, device):
    """
    Make predictions using a trained model.
    
    Args:
        model (torch.nn.Module): Trained model
        dataloader (torch.utils.data.DataLoader): Dataloader with test data
        device (torch.device): Device to evaluate on
        
    Returns:
        tuple: (predictions, true_labels, logits, inference_time)
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_logits = []
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # If outputs is a tuple (loss, logits), take logits
            logits = outputs[1] if isinstance(outputs, tuple) else outputs
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            
            # Get labels
            all_labels.extend(labels.cpu().numpy())
            
            # Get logits
            all_logits.extend(logits.cpu().numpy())
    
    # Calculate inference time
    inference_time = time.time() - start_time
    avg_inference_time = inference_time / len(dataloader.dataset)
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_logits), avg_inference_time


def evaluate_model(model, test_dataloader, device, model_type, output_dir=config.OUTPUT_DIR):
    """
    Evaluate model on test data.
    
    Args:
        model (torch.nn.Module): Trained model
        test_dataloader (torch.utils.data.DataLoader): Test dataloader
        device (torch.device): Device to evaluate on
        model_type (str): Type of model
        output_dir (str): Directory to save evaluation results
        
    Returns:
        dict: Evaluation results
    """
    print(f"Evaluating {model_type} model...")
    
    # Get predictions
    predictions, true_labels, logits, inference_time = predict(
        model, test_dataloader, device
    )
    
    # Get probabilities
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
    
    # Calculate metrics
    val_loss, metrics = evaluate(model, test_dataloader, device)
    
    # Create classification report
    report = classification_report(true_labels, predictions, output_dict=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Create evaluation results
    eval_results = {
        "model_type": model_type,
        "metrics": metrics,
        "val_loss": val_loss,
        "inference_time": inference_time,
        "predictions": predictions,
        "true_labels": true_labels,
        "probabilities": probabilities,
        "confusion_matrix": cm,
        "classification_report": report
    }
    
    # Create evaluation directory
    eval_dir = os.path.join(output_dir, "evaluation", model_type)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Save visualizations
    visualize_results(eval_results, eval_dir)
    
    return eval_results


def visualize_results(eval_results, output_dir):
    """
    Visualize evaluation results.
    
    Args:
        eval_results (dict): Evaluation results
        output_dir (str): Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    model_type = eval_results["model_type"]
    cm = eval_results["confusion_matrix"]
    true_labels = eval_results["true_labels"]
    probabilities = eval_results["probabilities"]
    predictions = eval_results["predictions"]
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Non-Complaint", "Complaint"],
                yticklabels=["Non-Complaint", "Complaint"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_type}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(true_labels, probabilities[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_type}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()
    
    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(true_labels, probabilities[:, 1])
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, lw=2, label=f"PR curve (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_type}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curve.png"))
    plt.close()
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        "Model": [model_type],
        "Accuracy": [eval_results["metrics"]["accuracy"]],
        "Precision": [eval_results["metrics"]["precision"]],
        "Recall": [eval_results["metrics"]["recall"]],
        "F1 Score": [eval_results["metrics"]["f1"]],
        "ROC AUC": [roc_auc],
        "PR AUC": [pr_auc],
        "Inference Time (ms)": [eval_results["inference_time"] * 1000]
    })
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)


def compare_models(eval_results_list, output_dir=config.OUTPUT_DIR):
    """
    Compare models based on evaluation results.
    
    Args:
        eval_results_list (list): List of evaluation results for different models
        output_dir (str): Directory to save comparison results
        
    Returns:
        tuple: (comparison_df, best_model_type)
    """
    print("Comparing models...")
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Create comparison dataframe
    comparison_data = []
    
    for eval_results in eval_results_list:
        model_type = eval_results["model_type"]
        metrics = eval_results["metrics"]
        inference_time = eval_results["inference_time"]
        
        # Calculate ROC AUC
        fpr, tpr, _ = roc_curve(eval_results["true_labels"], eval_results["probabilities"][:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Calculate PR AUC
        precision, recall, _ = precision_recall_curve(eval_results["true_labels"], eval_results["probabilities"][:, 1])
        pr_auc = auc(recall, precision)
        
        comparison_data.append({
            "Model Type": model_type,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1"],
            "ROC AUC": roc_auc,
            "PR AUC": pr_auc,
            "Inference Time (ms)": inference_time * 1000,
            "Data Type": "Full" if "full" in model_type else "Caller Only",
            "Architecture": model_type.split("_")[0] if "_" in model_type else model_type
        })
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by F1 score
    comparison_df = comparison_df.sort_values("F1 Score", ascending=False)
    
    # Save comparison table
    comparison_df.to_csv(os.path.join(comparison_dir, "model_comparison.csv"), index=False)
    
    # Get best model
    best_model_type = comparison_df.iloc[0]["Model Type"]
    
    # Plot comparison charts
    plot_comparison_charts(comparison_df, comparison_dir)
    
    # Print comparison table
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    print(f"\nBest Model: {best_model_type}")
    
    return comparison_df, best_model_type


def plot_comparison_charts(comparison_df, output_dir):
    """
    Plot comparison charts.
    
    Args:
        comparison_df (pd.DataFrame): Comparison dataframe
        output_dir (str): Directory to save charts
    """
    # Plot F1 score comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model Type", y="F1 Score", data=comparison_df)
    plt.title("F1 Score Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_comparison.png"))
    plt.close()
    
    # Plot inference time comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model Type", y="Inference Time (ms)", data=comparison_df)
    plt.title("Inference Time Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inference_time_comparison.png"))
    plt.close()
    
    # Plot metrics by data type
    plt.figure(figsize=(12, 6))
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    data_types = comparison_df["Data Type"].unique()
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        sns.barplot(x="Data Type", y=metric, data=comparison_df)
        plt.title(f"{metric} by Data Type")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_by_data_type.png"))
    plt.close()
    
    # Plot metrics by architecture
    plt.figure(figsize=(12, 6))
    architectures = comparison_df["Architecture"].unique()
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        sns.barplot(x="Architecture", y=metric, data=comparison_df)
        plt.title(f"{metric} by Architecture")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_by_architecture.png"))
    plt.close()


def evaluate_all_models(models, test_dataloaders, device, output_dir=config.OUTPUT_DIR):
    """
    Evaluate all models.
    
    Args:
        models (dict): Dictionary of trained models
        test_dataloaders (dict): Dictionary of test dataloaders
        device (torch.device): Device to evaluate on
        output_dir (str): Directory to save evaluation results
        
    Returns:
        tuple: (eval_results_list, comparison_df, best_model_type)
    """
    # Initialize evaluation results list
    eval_results_list = []
    
    # Evaluate each model
    for model_type, model in models.items():
        # Evaluate model
        eval_results = evaluate_model(
            model,
            test_dataloaders[model_type],
            device,
            model_type,
            output_dir
        )
        
        # Add to results list
        eval_results_list.append(eval_results)
    
    # Compare models
    comparison_df, best_model_type = compare_models(eval_results_list, output_dir)
    
    return eval_results_list, comparison_df, best_model_type 