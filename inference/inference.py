"""
Inference module for complaint detection.
Provides a pipeline for making predictions on new data.
"""

import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import sys
sys.path.append('.')
import config
from models.model_factory import load_model
from utils.data_preprocessing import clean_text, create_caller_only_version
from transformers import AutoTokenizer


class ComplaintDetector:
    """Class for detecting complaints in conversations."""
    
    def __init__(self, model_type=None, model_path=None, threshold=0.3):
        """
        Initialize complaint detector.
        
        Args:
            model_type (str, optional): Type of model to use
            model_path (str, optional): Path to model checkpoint
            threshold (float): Threshold for complaint classification (default: 0.3 for imbalanced data)
        """
        self.model_type = model_type
        self.model_path = model_path
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        if self.model_type is None or self.model_path is None:
            raise ValueError("Model type and model path must be specified")
        
        print(f"Loading model {self.model_type} from {self.model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
        
        # Load model
        self.model = load_model(self.model_type, self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Determine if model uses caller-only utterances
        self.use_caller_only = "caller_only" in self.model_type
        
        print("Model loaded successfully!")
        
    def preprocess_text(self, text):
        """
        Preprocess text for inference.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Clean text
        cleaned_text = clean_text(
            text,
            lower_case=config.LOWER_CASE,
            remove_special_chars=config.REMOVE_SPECIAL_CHARS
        )
        
        # Extract caller-only utterances if needed
        if self.use_caller_only:
            return create_caller_only_version(cleaned_text)
        
        return cleaned_text
        
    def tokenize_text(self, text):
        """
        Tokenize text for model input.
        
        Args:
            text (str): Preprocessed text
            
        Returns:
            dict: Tokenized input
        """
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        return encoding
        
    def predict(self, text):
        """
        Predict if text contains a complaint.
        
        Args:
            text (str): Input text
            
        Returns:
            tuple: (is_complaint, probability, logits)
        """
        # Preprocess text
        preprocessed_text = self.preprocess_text(text)
        
        # Tokenize text
        encoding = self.tokenize_text(preprocessed_text)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            
            # If outputs is a tuple (loss, logits), take logits
            logits = outputs[1] if isinstance(outputs, tuple) else outputs
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # Get complaint probability
            complaint_probability = probabilities[0, 1].item()
            
            # Classify as complaint if probability >= threshold
            is_complaint = complaint_probability >= self.threshold
            
        return bool(is_complaint), complaint_probability, logits.cpu().numpy()
        
    def predict_batch(self, texts, batch_size=config.INFERENCE_BATCH_SIZE):
        """
        Predict complaints in a batch of texts.
        
        Args:
            texts (list): List of input texts
            batch_size (int): Batch size for inference
            
        Returns:
            tuple: (is_complaint_list, probability_list, logits_list)
        """
        # Initialize result lists
        is_complaint_list = []
        probability_list = []
        logits_list = []
        
        # Preprocess texts
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        # Process in batches
        for i in tqdm(range(0, len(preprocessed_texts), batch_size), desc="Processing batches"):
            # Get batch
            batch_texts = preprocessed_texts[i:i+batch_size]
            
            # Tokenize batch
            batch_encodings = self.tokenizer(
                batch_texts,
                max_length=config.MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            batch_encodings = {k: v.to(self.device) for k, v in batch_encodings.items()}
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch_encodings['input_ids'],
                    attention_mask=batch_encodings['attention_mask']
                )
                
                # If outputs is a tuple (loss, logits), take logits
                logits = outputs[1] if isinstance(outputs, tuple) else outputs
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                
                # Get complaint probabilities
                batch_complaint_probabilities = probabilities[:, 1].cpu().numpy()
                
                # Classify as complaint if probability >= threshold
                batch_is_complaint = batch_complaint_probabilities >= self.threshold
                
                # Add to result lists
                is_complaint_list.extend(batch_is_complaint.tolist())
                probability_list.extend(batch_complaint_probabilities.tolist())
                logits_list.extend(logits.cpu().numpy())
        
        return is_complaint_list, probability_list, logits_list
        
    def calculate_complaint_percentage(self, texts):
        """
        Calculate percentage of complaints in a list of texts.
        
        Args:
            texts (list): List of input texts
            
        Returns:
            float: Percentage of complaints
        """
        # Predict complaints
        is_complaint_list, _, _ = self.predict_batch(texts)
        
        # Calculate percentage
        complaint_percentage = sum(is_complaint_list) / len(is_complaint_list)
        
        return complaint_percentage
        
    def analyze_conversation(self, conversation_df, text_column="text", timestamp_column=None):
        """
        Analyze a conversation dataframe for complaints.
        
        Args:
            conversation_df (pd.DataFrame): Dataframe with conversation data
            text_column (str): Column containing text
            timestamp_column (str, optional): Column containing timestamps
            
        Returns:
            pd.DataFrame: Dataframe with complaint analysis
        """
        # Get texts
        texts = conversation_df[text_column].tolist()
        
        # Predict complaints
        is_complaint_list, probability_list, _ = self.predict_batch(texts)
        
        # Add results to dataframe
        result_df = conversation_df.copy()
        result_df["is_complaint"] = is_complaint_list
        result_df["complaint_probability"] = probability_list
        
        # Calculate complaint percentage
        complaint_percentage = sum(is_complaint_list) / len(is_complaint_list)
        
        # Generate intensity values (binned probabilities)
        result_df["complaint_intensity"] = pd.cut(
            result_df["complaint_probability"],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["Low", "Moderate", "High", "Very High"]
        )
        
        print(f"Complaint Analysis: {sum(is_complaint_list)} complaints out of {len(is_complaint_list)} utterances")
        print(f"Complaint Percentage: {complaint_percentage:.2%}")
        
        return result_df, complaint_percentage


def load_best_model():
    """
    Load the best model from the comparison results.
    
    Returns:
        ComplaintDetector: Initialized complaint detector
    """
    # Find comparison csv
    comparison_path = os.path.join(config.OUTPUT_DIR, "comparison", "model_comparison.csv")
    
    # If comparison doesn't exist, return None
    if not os.path.exists(comparison_path):
        raise FileNotFoundError(f"Comparison file not found at {comparison_path}")
    
    # Load comparison dataframe
    comparison_df = pd.read_csv(comparison_path)
    
    # Get best model
    best_model_type = comparison_df.iloc[0]["Model Type"]
    
    # Find model path
    model_path = os.path.join(config.OUTPUT_DIR, f"{best_model_type}.pt")
    
    # If model doesn't exist, return None
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Initialize complaint detector
    detector = ComplaintDetector(
        model_type=best_model_type,
        model_path=model_path
    )
    
    return detector 