import os
import glob
import time
import json
from datetime import datetime

def get_latest_model_dir(output_dir="output"):
    """Get the most recently created model directory."""
    model_dirs = glob.glob(os.path.join(output_dir, "models_*"))
    if not model_dirs:
        return None
    
    # Sort by creation time (most recent first)
    model_dirs.sort(key=os.path.getctime, reverse=True)
    return model_dirs[0]

def get_model_files(model_dir):
    """Get a list of all model files in the directory."""
    if not os.path.exists(model_dir):
        return []
    
    files = os.listdir(model_dir)
    return files

def get_training_status(model_dir):
    """Get training status from available files."""
    files = get_model_files(model_dir)
    
    # Check if any model files exist
    model_files = [f for f in files if f.endswith('.pt')]
    if not model_files:
        return "Training in progress, no models saved yet"
    
    # Check which models have been completed
    completed_models = []
    for model_type in ['lstm_cnn_full', 'lstm_cnn_caller_only', 'mlm_full', 'mlm_caller_only']:
        if f"{model_type}.pt" in files:
            completed_models.append(model_type)
    
    # Check for metadata to get training details
    metrics = {}
    for model in completed_models:
        metadata_file = os.path.join(model_dir, f"{model}_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                try:
                    metadata = json.load(f)
                    if 'test_metrics' in metadata:
                        metrics[model] = metadata['test_metrics']
                except:
                    pass
    
    return {
        "completed_models": completed_models,
        "total_models": 4,
        "progress": f"{len(completed_models)}/4 models completed",
        "metrics": metrics
    }

def main():
    """Main function to check training progress."""
    model_dir = get_latest_model_dir()
    if not model_dir:
        print("No model directories found.")
        return
    
    print(f"Checking progress in: {model_dir}")
    
    # Initial check
    status = get_training_status(model_dir)
    if isinstance(status, str):
        print(status)
        return
    
    print(f"Progress: {status['progress']}")
    print("Completed models:")
    for model in status['completed_models']:
        print(f"  - {model}")
        if model in status['metrics']:
            metrics = status['metrics'][model]
            print(f"    Accuracy: {metrics.get('accuracy', 'N/A')}")
            print(f"    F1 Score: {metrics.get('f1_score', 'N/A')}")
    
    # Check for comparison file
    comparison_file = os.path.join(model_dir, "model_comparison.csv")
    if os.path.exists(comparison_file):
        print("\nTraining and evaluation complete!")
        print(f"See comparison results in: {comparison_file}")

if __name__ == "__main__":
    main() 