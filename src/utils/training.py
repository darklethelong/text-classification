import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import time

def train_epoch(model, data_loader, optimizer, criterion, device):
    """Train model for one epoch."""
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    # Use tqdm for progress bar
    pbar = tqdm(data_loader, desc="Training")
    
    for batch in pbar:
        # Move data to device
        if hasattr(batch, 'keys') and 'input_ids' in batch and 'attention_mask' in batch:
            # For MLM models
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            # For LSTM-CNN models
            indices = batch['indices'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(indices)
        
        # Calculate loss
        loss = criterion(logits, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Update metrics
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'acc': acc.item()})
    
    # Return average loss and accuracy
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)

def evaluate(model, data_loader, criterion, device):
    """Evaluate model on validation or test data."""
    model.eval()
    epoch_loss = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            if hasattr(batch, 'keys') and 'input_ids' in batch and 'attention_mask' in batch:
                # For MLM models
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                # For LSTM-CNN models
                indices = batch['indices'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                logits = model(indices)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Calculate predictions
            preds = torch.argmax(logits, dim=1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update loss
            epoch_loss += loss.item()
    
    # Convert to numpy arrays for sklearn metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Return metrics
    return {
        'loss': epoch_loss / len(data_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
               device, num_epochs=10, early_stopping_patience=3, model_save_path=None):
    """Train model with early stopping."""
    
    # Track best validation F1 score
    best_val_f1 = 0
    best_model = None
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': []
    }
    
    # Move model to device
    model.to(device)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        train_time = time.time() - start_time
        
        # Validation phase
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate based on validation loss
        scheduler.step(val_metrics['loss'])
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f} | Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
        print(f"Training time: {train_time:.2f}s")
        print("-" * 50)
        
        # Check for improvement
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model = model.state_dict().copy()
            patience_counter = 0
            
            # Save model if path is provided
            if model_save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'history': history
                }, model_save_path)
                print(f"Model saved to {model_save_path}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model if available
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model, history

def measure_inference_latency_cpu(model, data_loader, num_samples=100):
    """Measure model inference latency on CPU."""
    # Force CPU
    cpu_device = torch.device('cpu')
    model = model.to(cpu_device)
    model.eval()
    latencies = []
    
    with torch.no_grad():
        # Limit to num_samples batches for consistent measurement
        for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break
                
            # Move data to CPU
            if hasattr(batch, 'keys') and 'input_ids' in batch and 'attention_mask' in batch:
                # For MLM models
                input_ids = batch['input_ids'].to(cpu_device)
                attention_mask = batch['attention_mask'].to(cpu_device)
                
                # Measure inference time
                start_time = time.time()
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                end_time = time.time()
            else:
                # For LSTM-CNN models
                indices = batch['indices'].to(cpu_device)
                
                # Measure inference time
                start_time = time.time()
                _ = model(indices)
                end_time = time.time()
            
            latencies.append(end_time - start_time)
    
    # Calculate statistics
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    return {
        'device': 'cpu',
        'avg_latency': avg_latency,
        'std_latency': std_latency,
        'p95_latency': p95_latency,
        'raw_latencies': latencies
    }

def measure_inference_latency_gpu(model, data_loader, num_samples=100):
    """Measure model inference latency on GPU, if available."""
    # Check if GPU is available
    if torch.cuda.is_available():
        gpu_device = torch.device('cuda')
        model = model.to(gpu_device)
    else:
        print("WARNING: GPU not available, falling back to CPU for measurement")
        return measure_inference_latency_cpu(model, data_loader, num_samples)
    
    model.eval()
    latencies = []
    
    # Warm-up the GPU
    print("Warming up GPU...")
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= 5:  # 5 warm-up batches
                break
                
            if hasattr(batch, 'keys') and 'input_ids' in batch and 'attention_mask' in batch:
                input_ids = batch['input_ids'].to(gpu_device)
                attention_mask = batch['attention_mask'].to(gpu_device)
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                indices = batch['indices'].to(gpu_device)
                _ = model(indices)
    
    # Actual measurement
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break
                
            # Move data to GPU
            if hasattr(batch, 'keys') and 'input_ids' in batch and 'attention_mask' in batch:
                # For MLM models
                input_ids = batch['input_ids'].to(gpu_device)
                attention_mask = batch['attention_mask'].to(gpu_device)
                
                # Ensure all previous GPU operations are completed
                torch.cuda.synchronize()
                
                # Measure inference time
                start_time = time.time()
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                torch.cuda.synchronize() # Wait for GPU operations to complete
                end_time = time.time()
            else:
                # For LSTM-CNN models
                indices = batch['indices'].to(gpu_device)
                
                # Ensure all previous GPU operations are completed
                torch.cuda.synchronize()
                
                # Measure inference time
                start_time = time.time()
                _ = model(indices)
                torch.cuda.synchronize() # Wait for GPU operations to complete
                end_time = time.time()
            
            latencies.append(end_time - start_time)
    
    # Calculate statistics
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    return {
        'device': 'gpu',
        'avg_latency': avg_latency,
        'std_latency': std_latency,
        'p95_latency': p95_latency,
        'raw_latencies': latencies
    }

def measure_inference_latency(model, data_loader, device, num_samples=100):
    """Measure model inference latency on the specified device."""
    # Call the appropriate specialized function based on the device
    if device.type == 'cuda':
        return measure_inference_latency_gpu(model, data_loader, num_samples)
    else:
        return measure_inference_latency_cpu(model, data_loader, num_samples) 