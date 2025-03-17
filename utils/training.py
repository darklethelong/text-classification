"""
Training module for complaint detection models.
Provides functions for training, evaluation, and metrics calculation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import time

import sys
sys.path.append('.')
import config
from models.model_factory import save_model


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=config.EARLY_STOPPING_PATIENCE, min_delta=0.001):
        """
        Initialize early stopping.
        
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change in monitored value to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        """
        Check if training should stop.
        
        Args:
            val_loss (float): Validation loss
            
        Returns:
            bool: Whether to stop training
        """
        if self.best_score is None:
            self.best_score = val_loss
            return False
        
        if val_loss >= self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = val_loss
            self.counter = 0
            
        return False


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
    
    return metrics


def train_epoch(model, dataloader, optimizer, device, scheduler=None):
    """
    Train model for one epoch.
    
    Args:
        model (torch.nn.Module): Model to train
        dataloader (torch.utils.data.DataLoader): Training dataloader
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to train on
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler
        
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        loss, logits = model(input_ids, attention_mask, labels)
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Update total loss
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": loss.item(),
            "accuracy": correct / total
        })
    
    # Calculate epoch metrics
    epoch_loss = total_loss / len(dataloader)
    epoch_accuracy = correct / total
    
    return epoch_loss, epoch_accuracy


def evaluate(model, dataloader, device):
    """
    Evaluate model.
    
    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (torch.utils.data.DataLoader): Evaluation dataloader
        device (torch.device): Device to evaluate on
        
    Returns:
        tuple: (val_loss, metrics)
    """
    model.eval()
    
    total_loss = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            loss, logits = model(input_ids, attention_mask, labels)
            
            # Calculate loss
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            
            # Get labels
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate validation loss
    val_loss = total_loss / len(dataloader)
    
    # Calculate metrics
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_predictions)
    )
    
    return val_loss, metrics


def train_model(model, train_dataloader, val_dataloader, model_type, device,
                num_epochs=config.NUM_EPOCHS, learning_rate=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY, save_dir=config.OUTPUT_DIR):
    """
    Train and evaluate model.
    
    Args:
        model (torch.nn.Module): Model to train
        train_dataloader (torch.utils.data.DataLoader): Training dataloader
        val_dataloader (torch.utils.data.DataLoader): Validation dataloader
        model_type (str): Type of model (for saving)
        device (torch.device): Device to train on
        num_epochs (int): Number of epochs to train for
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay
        save_dir (str): Directory to save model in
        
    Returns:
        dict: Training history
    """
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Initialize learning rate scheduler
    total_steps = len(train_dataloader) * num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(os.path.join(save_dir, "logs", model_type))
    
    # Initialize training history
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_metrics": []
    }
    
    # Train model
    print(f"Starting training for {model_type}...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train epoch
        train_loss, train_accuracy = train_epoch(
            model, train_dataloader, optimizer, device, scheduler
        )
        
        # Evaluate
        val_loss, val_metrics = evaluate(model, val_dataloader, device)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        # Update history
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)
        
        # Log to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        for metric_name, metric_value in val_metrics.items():
            writer.add_scalar(f"{metric_name.capitalize()}/val", metric_value, epoch)
        
        # Save model (every few epochs or at the end)
        if (epoch + 1) % config.SAVE_STEPS == 0 or epoch == num_epochs - 1:
            save_path = save_model(model, model_type, save_dir, epoch=(epoch + 1))
            print(f"Model saved to {save_path}")
        
        # Check for early stopping
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Calculate total training time
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    # Save final model
    save_path = save_model(model, model_type, save_dir)
    print(f"Final model saved to {save_path}")
    
    # Close tensorboard writer
    writer.close()
    
    return history


def train_all_models(train_dataloaders, val_dataloaders, vocab_size=30000, device=None):
    """
    Train all models.
    
    Args:
        train_dataloaders (dict): Dictionary of training dataloaders for each model type
        val_dataloaders (dict): Dictionary of validation dataloaders for each model type
        vocab_size (int): Vocabulary size for LSTM-CNN models
        device (torch.device, optional): Device to train on
        
    Returns:
        dict: Training history for each model
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Initialize history
    all_history = {}
    
    # Train each model
    for model_type in config.MODELS_TO_TRAIN:
        print(f"\n{'='*50}")
        print(f"Training {model_type} model")
        print(f"{'='*50}\n")
        
        # Get model
        from models.model_factory import get_model
        model = get_model(model_type, vocab_size)
        model.to(device)
        
        # Train model
        history = train_model(
            model,
            train_dataloaders[model_type],
            val_dataloaders[model_type],
            model_type,
            device
        )
        
        # Save history
        all_history[model_type] = history
    
    return all_history 