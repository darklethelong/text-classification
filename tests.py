"""
Test script for the complaint detection system.
Run this to ensure that all components are working properly.
"""

import os
import torch
import pandas as pd
import unittest
from tqdm import tqdm

import config
from utils.data_preprocessing import load_and_preprocess_data, split_data
from models.model_factory import get_model
from models.mlm_models import MLMComplaintDetector
from models.lstm_cnn_models import LSTMCNNComplaintDetector
from models.hybrid_models import HybridComplaintDetector
from inference.inference import ComplaintDetector


class TestComplaintDetection(unittest.TestCase):
    """Test class for the complaint detection system."""
    
    def setUp(self):
        """Set up tests."""
        self.sample_data_path = os.path.join("data", "sample_data.csv")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure sample data exists
        if not os.path.exists(self.sample_data_path):
            raise FileNotFoundError(f"Sample data not found at {self.sample_data_path}")
    
    def test_data_processing(self):
        """Test data processing functions."""
        print("\nTesting data preprocessing...")
        
        # Load and preprocess data
        df = load_and_preprocess_data(self.sample_data_path)
        
        # Check that required columns exist
        required_columns = ['text', 'label', 'cleaned_text', 'caller_only_text']
        for col in required_columns:
            self.assertIn(col, df.columns, f"Missing column: {col}")
        
        # Check that no values are null
        for col in required_columns:
            self.assertFalse(df[col].isnull().any(), f"Null values found in column: {col}")
        
        # Split data
        train_df, val_df, test_df = split_data(df)
        
        # Check that split was performed correctly
        total_rows = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(total_rows, len(df), "Data split doesn't match original size")
        
        print("Data preprocessing tests passed!")
    
    def test_model_creation(self):
        """Test model creation."""
        print("\nTesting model creation...")
        
        # Test MLM model
        mlm_model = get_model("mlm_full")
        self.assertIsInstance(mlm_model, MLMComplaintDetector, "MLM model not created correctly")
        
        # Test LSTM-CNN model
        lstm_cnn_model = get_model("lstm_cnn_full", vocab_size=1000)
        self.assertIsInstance(lstm_cnn_model, LSTMCNNComplaintDetector, "LSTM-CNN model not created correctly")
        
        # Test Hybrid model
        hybrid_model = get_model("hybrid_full")
        self.assertIsInstance(hybrid_model, HybridComplaintDetector, "Hybrid model not created correctly")
        
        print("Model creation tests passed!")
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        print("\nTesting model forward pass...")
        
        # Create a small batch of data
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(0, 2, (batch_size,))
        
        # Test each model type
        model_types = ["mlm_full", "lstm_cnn_full", "hybrid_full"]
        
        for model_type in model_types:
            print(f"Testing forward pass for {model_type}...")
            
            # Get model (use small vocab for LSTM-CNN)
            vocab_size = 1000 if model_type == "lstm_cnn_full" else None
            model = get_model(model_type, vocab_size=vocab_size)
            model.to(self.device)
            
            # Move data to device
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with labels
            loss, logits = model(input_ids, attention_mask, labels)
            
            # Check output shapes
            self.assertEqual(logits.shape, (batch_size, 2), f"Incorrect logits shape for {model_type}")
            self.assertIsInstance(loss.item(), float, f"Loss is not a float for {model_type}")
            
            # Forward pass without labels
            logits = model(input_ids, attention_mask)
            
            # Check output shape
            self.assertEqual(logits.shape, (batch_size, 2), f"Incorrect logits shape for {model_type}")
        
        print("Model forward pass tests passed!")
    
    def test_inference(self):
        """Test inference pipeline."""
        print("\nTesting inference pipeline...")
        
        # Create and save a dummy model for testing
        os.makedirs("models/test", exist_ok=True)
        model_type = "mlm_full"
        model_path = os.path.join("models", "test", "test_model.pt")
        
        model = get_model(model_type)
        torch.save(model.state_dict(), model_path)
        
        try:
            # Initialize complaint detector
            detector = ComplaintDetector(
                model_type=model_type,
                model_path=model_path,
                threshold=0.5
            )
            
            # Test single prediction
            test_text = "Agent: How can I help you? Caller: I'm extremely frustrated with your service. I've been waiting for a refund for weeks!"
            is_complaint, probability, _ = detector.predict(test_text)
            
            # Check that prediction is a boolean and probability is a float
            self.assertIsInstance(is_complaint, bool, "Prediction is not a boolean")
            self.assertIsInstance(probability, float, "Probability is not a float")
            self.assertTrue(0 <= probability <= 1, "Probability not in range [0, 1]")
            
            # Test batch prediction
            df = pd.read_csv(self.sample_data_path)
            texts = df["text"].tolist()[:3]  # Take first 3 texts
            
            is_complaint_list, probability_list, _ = detector.predict_batch(texts)
            
            # Check that we get the expected number of predictions
            self.assertEqual(len(is_complaint_list), len(texts), "Number of predictions doesn't match number of texts")
            self.assertEqual(len(probability_list), len(texts), "Number of probabilities doesn't match number of texts")
            
            print("Inference pipeline tests passed!")
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.remove(model_path)
    
    def test_all_components(self):
        """Run all tests."""
        self.test_data_processing()
        self.test_model_creation()
        self.test_model_forward_pass()
        self.test_inference()
        
        print("\nAll tests passed! The complaint detection system is working properly.")


if __name__ == "__main__":
    unittest.main() 