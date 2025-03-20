import os
import argparse
import sys
from datetime import datetime
import traceback

def main():
    """
    Main function to run the complete workflow.
    """
    try:
        print("Starting Customer Complaint Detection System")
        print("Python version:", sys.version)
        
        parser = argparse.ArgumentParser(description='Customer Complaint Detection System')
        parser.add_argument('--data_file', type=str, default='train.csv', 
                            help='Path to the data file (default: train.csv)')
        parser.add_argument('--output_dir', type=str, default='output', 
                            help='Output directory for models and results (default: output)')
        parser.add_argument('--models', type=str, default='all', 
                            choices=['all', 'lstm_cnn_full', 'lstm_cnn_caller_only', 'mlm_full', 'mlm_caller_only'],
                            help='Which models to train (default: all)')
        parser.add_argument('--skip_training', action='store_true', 
                            help='Skip training and use existing models')
        parser.add_argument('--model_path', type=str, 
                            help='Path to the trained model file (.pt) for inference (required if skip_training=True)')
        parser.add_argument('--model_type', type=str, choices=['lstm-cnn', 'mlm'], 
                            help='Type of model for inference (required if skip_training=True)')
        parser.add_argument('--caller_only', action='store_true', 
                            help='Use only caller utterances for inference')
        parser.add_argument('--input_file', type=str, 
                            help='Path to input conversation file for inference')
        args = parser.parse_args()
        
        print(f"Arguments: {args}")
        print(f"Current working directory: {os.getcwd()}")
        
        # Verify the data file exists
        if not os.path.exists(args.data_file):
            print(f"ERROR: Data file {args.data_file} does not exist!")
            return
        else:
            print(f"Data file {args.data_file} found. Size: {os.path.getsize(args.data_file)} bytes")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory {args.output_dir} is ready")
        
        # Training phase
        if not args.skip_training:
            print("="*80)
            print("TRAINING PHASE")
            print("="*80)
            
            # Check if src directory and required modules exist
            if not os.path.exists('src') or not os.path.exists('src/train_and_evaluate.py'):
                print("ERROR: Required source files not found. Please make sure the project structure is correct.")
                print(f"Existing files in current directory: {os.listdir('.')}")
                if os.path.exists('src'):
                    print(f"Files in src directory: {os.listdir('src')}")
                return
            
            # Run training script
            train_cmd = f"python src/train_and_evaluate.py --data_file {args.data_file} --output_dir {args.output_dir} --models {args.models}"
            print(f"Running command: {train_cmd}")
            exit_code = os.system(train_cmd)
            
            if exit_code != 0:
                print(f"WARNING: Training command exited with code {exit_code}")
            
            # Get the latest model directory
            model_dirs = [os.path.join(args.output_dir, d) for d in os.listdir(args.output_dir) 
                        if os.path.isdir(os.path.join(args.output_dir, d)) and d.startswith('models_')]
            
            print(f"Available model directories: {model_dirs}")
            
            if model_dirs:
                latest_model_dir = max(model_dirs, key=os.path.getmtime)
                print(f"Using latest model directory: {latest_model_dir}")
                
                # Find the best model based on model type
                if args.models == 'all':
                    # Default to MLM full for best performance
                    model_path = os.path.join(latest_model_dir, "mlm_full.pt")
                    model_type = "mlm"
                elif args.models.startswith('lstm_cnn'):
                    model_path = os.path.join(latest_model_dir, f"{args.models}.pt")
                    model_type = "lstm-cnn"
                elif args.models.startswith('mlm'):
                    model_path = os.path.join(latest_model_dir, f"{args.models}.pt")
                    model_type = "mlm"
                else:
                    raise ValueError(f"Unknown model type: {args.models}")
                
                # Check if model file exists
                if not os.path.exists(model_path):
                    print(f"ERROR: Model file {model_path} not found!")
                    print(f"Available files in {latest_model_dir}: {os.listdir(latest_model_dir)}")
                    return
                
                print(f"Selected model: {model_path} (type: {model_type})")
                
                # Use caller_only flag based on the trained model
                caller_only = "caller_only" in args.models
            else:
                print("No trained model found. Please provide model_path and model_type for inference.")
                return
        else:
            # Use provided model path and type
            if not args.model_path or not args.model_type:
                print("Error: --model_path and --model_type are required when using --skip_training")
                return
            
            model_path = args.model_path
            model_type = args.model_type
            caller_only = args.caller_only
            
            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"ERROR: Model file {model_path} not found!")
                return
        
        # Inference phase
        print("\n" + "="*80)
        print("INFERENCE PHASE")
        print("="*80)
        
        # Check if inference script exists
        if not os.path.exists('src/inference_demo.py'):
            print("ERROR: Required inference script not found.")
            return
        
        # Build inference command
        inference_cmd = f"python src/inference_demo.py --model_path {model_path} --model_type {model_type}"
        
        if caller_only:
            inference_cmd += " --caller_only"
        
        if args.input_file:
            if not os.path.exists(args.input_file):
                print(f"WARNING: Input file {args.input_file} not found!")
            else:
                inference_cmd += f" --input_file {args.input_file}"
        
        # Run inference script
        print(f"Running command: {inference_cmd}")
        exit_code = os.system(inference_cmd)
        
        if exit_code != 0:
            print(f"WARNING: Inference command exited with code {exit_code}")
        
        print("="*80)
        print("EXECUTION COMPLETED")
        print("="*80)
        
    except Exception as e:
        print("="*80)
        print("ERROR OCCURRED DURING EXECUTION:")
        print(str(e))
        print("-"*80)
        print("Traceback:")
        traceback.print_exc()
        print("="*80)

if __name__ == "__main__":
    main() 