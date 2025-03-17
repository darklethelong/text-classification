"""
Main script for the complaint detection system.
Provides the entry point for training, evaluating, and using the models.
"""

import os
import argparse
import torch
import pandas as pd
from datetime import datetime

# Import project modules
import config
from utils.data_preprocessing import prepare_data_pipeline
from models.model_factory import get_model, save_model, load_model
from utils.training import train_model, train_all_models
from utils.evaluation import evaluate_model, evaluate_all_models
from inference.inference import ComplaintDetector, load_best_model
from visualization.visualization import generate_dashboard, generate_report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Complaint Detection System")
    
    # Main command
    parser.add_argument("command", choices=["train", "evaluate", "predict", "visualize"],
                       help="Main command to execute")
    
    # Common arguments
    parser.add_argument("--data_path", type=str, default=config.DATA_PATH,
                       help="Path to the dataset file")
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR,
                       help="Directory to save outputs")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda or cpu)")
    
    # Train arguments
    parser.add_argument("--model_type", type=str, choices=config.MODELS_TO_TRAIN,
                       default="mlm_full", help="Type of model to train")
    parser.add_argument("--num_epochs", type=int, default=config.NUM_EPOCHS,
                       help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE,
                       help="Learning rate for training")
    parser.add_argument("--train_all", action="store_true",
                       help="Train all model types")
    
    # Evaluate arguments
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--evaluate_all", action="store_true",
                       help="Evaluate all trained models")
    
    # Predict arguments
    parser.add_argument("--input_file", type=str, default=None,
                       help="Path to input file for prediction")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Path to output file for prediction results")
    parser.add_argument("--text_column", type=str, default=config.TEXT_COLUMN,
                       help="Column containing text data")
    parser.add_argument("--threshold", type=float, default=config.COMPLAINT_THRESHOLD,
                       help="Threshold for complaint classification")
    
    # Visualize arguments
    parser.add_argument("--timestamp_column", type=str, default=None,
                       help="Column containing timestamps")
    parser.add_argument("--generate_report", action="store_true",
                       help="Generate HTML report")
    
    return parser.parse_args()


def train(args):
    """
    Train models.
    
    Args:
        args: Command line arguments
    """
    print("Starting training...")
    
    # Set device
    device = args.device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train all models
    if args.train_all:
        print("Training all model types...")
        
        # Prepare dataloaders for all models
        dataloaders = {}
        tokenizers = {}
        
        for model_type in config.MODELS_TO_TRAIN:
            use_caller_only = "caller_only" in model_type
            print(f"Preparing data for {model_type} (caller_only={use_caller_only})...")
            
            train_dataloader, val_dataloader, test_dataloader, tokenizer = prepare_data_pipeline(
                use_caller_only=use_caller_only
            )
            
            dataloaders[model_type] = {
                "train": train_dataloader,
                "val": val_dataloader,
                "test": test_dataloader
            }
            tokenizers[model_type] = tokenizer
        
        # Train all models
        train_dataloaders = {model_type: data["train"] for model_type, data in dataloaders.items()}
        val_dataloaders = {model_type: data["val"] for model_type, data in dataloaders.items()}
        
        all_history = train_all_models(train_dataloaders, val_dataloaders, device=device)
        
        # Save training history
        history_file = os.path.join(args.output_dir, "training_history.pkl")
        pd.to_pickle(all_history, history_file)
        print(f"Training history saved to {history_file}")
        
    # Train a single model
    else:
        print(f"Training {args.model_type} model...")
        
        # Prepare data
        use_caller_only = "caller_only" in args.model_type
        train_dataloader, val_dataloader, test_dataloader, tokenizer = prepare_data_pipeline(
            use_caller_only=use_caller_only
        )
        
        # Get model
        model = get_model(args.model_type)
        model.to(device)
        
        # Train model
        history = train_model(
            model,
            train_dataloader,
            val_dataloader,
            args.model_type,
            device,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            save_dir=args.output_dir
        )
        
        # Save training history
        history_file = os.path.join(args.output_dir, f"{args.model_type}_history.pkl")
        pd.to_pickle(history, history_file)
        print(f"Training history saved to {history_file}")
    
    print("Training complete!")


def evaluate(args):
    """
    Evaluate models.
    
    Args:
        args: Command line arguments
    """
    print("Starting evaluation...")
    
    # Set device
    device = args.device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate all models
    if args.evaluate_all:
        print("Evaluating all trained models...")
        
        # Find all model files
        model_files = [f for f in os.listdir(args.output_dir) if f.endswith(".pt")]
        
        if not model_files:
            print("No trained models found in output directory.")
            return
        
        # Prepare data for all models
        dataloaders = {}
        models = {}
        
        for model_file in model_files:
            model_type = model_file.split(".")[0]
            model_path = os.path.join(args.output_dir, model_file)
            
            print(f"Loading model {model_type} from {model_path}...")
            
            # Load model
            model = load_model(model_type, model_path)
            model.to(device)
            models[model_type] = model
            
            # Prepare data
            use_caller_only = "caller_only" in model_type
            print(f"Preparing data for {model_type} (caller_only={use_caller_only})...")
            
            train_dataloader, val_dataloader, test_dataloader, _ = prepare_data_pipeline(
                use_caller_only=use_caller_only
            )
            
            dataloaders[model_type] = test_dataloader
        
        # Evaluate all models
        eval_results, comparison_df, best_model = evaluate_all_models(
            models, dataloaders, device, args.output_dir
        )
        
        print(f"Best model: {best_model}")
        
    # Evaluate a single model
    else:
        if args.model_path is None or args.model_type is None:
            print("Error: Both --model_path and --model_type must be specified.")
            return
        
        print(f"Evaluating {args.model_type} model...")
        
        # Load model
        model = load_model(args.model_type, args.model_path)
        model.to(device)
        
        # Prepare data
        use_caller_only = "caller_only" in args.model_type
        _, _, test_dataloader, _ = prepare_data_pipeline(
            use_caller_only=use_caller_only
        )
        
        # Evaluate model
        eval_results = evaluate_model(
            model,
            test_dataloader,
            device,
            args.model_type,
            args.output_dir
        )
        
        print("Evaluation results:")
        for metric_name, metric_value in eval_results["metrics"].items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        print(f"Inference time: {eval_results['inference_time']*1000:.2f} ms per sample")
    
    print("Evaluation complete!")


def predict(args):
    """
    Make predictions on new data.
    
    Args:
        args: Command line arguments
    """
    print("Starting prediction...")
    
    if args.input_file is None:
        print("Error: --input_file must be specified.")
        return
    
    # Determine output file
    output_file = args.output_file
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        output_file = os.path.join(args.output_dir, f"{base_name}_predictions.csv")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load the model
    if args.model_path is not None and args.model_type is not None:
        # Use specified model
        print(f"Loading model {args.model_type} from {args.model_path}...")
        detector = ComplaintDetector(
            model_type=args.model_type,
            model_path=args.model_path,
            threshold=args.threshold
        )
    else:
        # Try to load best model
        try:
            print("Loading best model from comparison results...")
            detector = load_best_model()
        except Exception as e:
            print(f"Error loading best model: {e}")
            print("Please specify --model_path and --model_type.")
            return
    
    # Load input data
    print(f"Loading data from {args.input_file}...")
    try:
        input_df = pd.read_csv(args.input_file)
    except Exception as e:
        print(f"Error loading input file: {e}")
        return
    
    # Check if text column exists
    if args.text_column not in input_df.columns:
        print(f"Error: Text column '{args.text_column}' not found in input file.")
        return
    
    # Analyze conversation
    print("Analyzing conversation...")
    result_df, complaint_percentage = detector.analyze_conversation(
        input_df,
        text_column=args.text_column,
        timestamp_column=args.timestamp_column
    )
    
    # Save results
    print(f"Saving results to {output_file}...")
    result_df.to_csv(output_file, index=False)
    
    # Visualize results if requested
    if args.generate_report:
        visualize(args, result_df)
    
    print("Prediction complete!")


def visualize(args, input_df=None):
    """
    Visualize prediction results.
    
    Args:
        args: Command line arguments
        input_df (pd.DataFrame, optional): Input dataframe with predictions
    """
    print("Starting visualization...")
    
    # Load input data if not provided
    if input_df is None:
        if args.input_file is None:
            print("Error: --input_file must be specified.")
            return
        
        print(f"Loading data from {args.input_file}...")
        try:
            input_df = pd.read_csv(args.input_file)
        except Exception as e:
            print(f"Error loading input file: {e}")
            return
        
        # Check required columns
        required_cols = ["is_complaint", "complaint_probability", "complaint_intensity"]
        missing_cols = [col for col in required_cols if col not in input_df.columns]
        
        if missing_cols:
            print(f"Error: Input file is missing required columns: {', '.join(missing_cols)}")
            print("Make sure to run prediction first or provide a file with prediction results.")
            return
    
    # Determine output directory
    vis_dir = os.path.join(args.output_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Generate dashboard
    print("Generating visualizations...")
    generate_dashboard(
        input_df,
        vis_dir,
        timestamp_column=args.timestamp_column,
        text_column=args.text_column
    )
    
    # Generate report if requested
    if args.generate_report:
        report_file = os.path.join(args.output_dir, "complaint_report.html")
        print(f"Generating report at {report_file}...")
        generate_report(
            input_df,
            report_file,
            timestamp_column=args.timestamp_column,
            text_column=args.text_column
        )
    
    print("Visualization complete!")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Execute command
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "visualize":
        visualize(args)
    else:
        print(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main() 