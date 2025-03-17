"""
Data preprocessing module for the complaint detection project.
Includes functions for loading, cleaning, and preparing data for training.
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

import sys
sys.path.append('.')
import config

class ComplaintDataset(Dataset):
    """PyTorch dataset for complaint detection."""
    
    def __init__(self, texts, labels, tokenizer, max_length=1024):
        """
        Initialize dataset.
        
        Args:
            texts (list): List of text strings
            labels (list): List of binary labels (0 or 1)
            tokenizer: Tokenizer for encoding text
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def clean_text(text, lower_case=True, remove_special_chars=True):
    """
    Clean text data.
    
    Args:
        text (str): Text to clean
        lower_case (bool): Whether to convert to lowercase
        remove_special_chars (bool): Whether to remove special characters
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    if lower_case:
        text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters
    if remove_special_chars:
        text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def create_caller_only_version(text):
    """
    Extract only the caller utterances from the conversation.
    Assuming format where caller utterances are identified by "Caller:" tag.
    
    Args:
        text (str): Full conversation text
        
    Returns:
        str: Text containing only caller utterances
    """
    # This is a simplified implementation and would need to be adapted
    # to the actual format of the conversation data
    lines = text.split('\n')
    caller_lines = []
    
    for line in lines:
        if line.lower().startswith('caller:'):
            caller_text = line.split(':', 1)[1].strip()
            caller_lines.append(caller_text)
    
    return ' '.join(caller_lines) if caller_lines else text


def load_and_preprocess_data(data_path=config.DATA_PATH, create_caller_only=True, df=None):
    """
    Load and preprocess the complaint dataset.
    
    Args:
        data_path (str): Path to the dataset file
        create_caller_only (bool): Whether to create a caller-only version
        df (pd.DataFrame, optional): DataFrame to use instead of loading from file
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    if df is None:
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print("Using provided DataFrame...")
    
    if config.TEXT_COLUMN not in df.columns or config.LABEL_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain '{config.TEXT_COLUMN}' and '{config.LABEL_COLUMN}' columns")
    
    # Clean text
    print("Cleaning text data...")
    df['cleaned_text'] = df[config.TEXT_COLUMN].apply(
        lambda x: clean_text(x, lower_case=config.LOWER_CASE, remove_special_chars=config.REMOVE_SPECIAL_CHARS)
    )
    
    # Create caller-only version if required
    if create_caller_only:
        print("Creating caller-only version of conversations...")
        df['caller_only_text'] = df['cleaned_text'].apply(create_caller_only_version)
    
    print(f"Data preprocessing complete. Found {len(df)} samples.")
    
    return df


def split_data(df, random_state=config.RANDOM_SEED):
    """
    Split data into training, validation, and test sets.
    
    Args:
        df (pd.DataFrame): Input dataframe
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # First split off the test set
    train_val_df, test_df = train_test_split(
        df, 
        test_size=config.TEST_RATIO,
        random_state=random_state,
        stratify=df[config.LABEL_COLUMN]
    )
    
    # Then split the remaining data into train and validation
    # Adjust the validation ratio to account for the already removed test set
    adjusted_val_ratio = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_ratio,
        random_state=random_state,
        stratify=train_val_df[config.LABEL_COLUMN]
    )
    
    print(f"Data split: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test samples")
    return train_df, val_df, test_df


def prepare_dataloaders(df_train, df_val, df_test, tokenizer, batch_size=config.BATCH_SIZE, use_caller_only=False):
    """
    Prepare DataLoader objects for training, validation, and testing.
    
    Args:
        df_train (pd.DataFrame): Training dataframe
        df_val (pd.DataFrame): Validation dataframe
        df_test (pd.DataFrame): Test dataframe
        tokenizer: Tokenizer for encoding text
        batch_size (int): Batch size
        use_caller_only (bool): Whether to use caller-only text
        
    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    # Select appropriate text column
    text_column = 'caller_only_text' if use_caller_only and 'caller_only_text' in df_train.columns else 'cleaned_text'
    
    # Create datasets
    train_dataset = ComplaintDataset(
        df_train[text_column].tolist(),
        df_train[config.LABEL_COLUMN].tolist(),
        tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    val_dataset = ComplaintDataset(
        df_val[text_column].tolist(),
        df_val[config.LABEL_COLUMN].tolist(),
        tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    test_dataset = ComplaintDataset(
        df_test[text_column].tolist(),
        df_test[config.LABEL_COLUMN].tolist(),
        tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_dataloader, val_dataloader, test_dataloader


def prepare_data_pipeline(use_caller_only=False, data_df=None, batch_size=config.BATCH_SIZE):
    """
    Complete data preparation pipeline.
    
    Args:
        use_caller_only (bool): Whether to use caller-only text
        data_df (pd.DataFrame, optional): Custom dataframe to use instead of loading from file
        batch_size (int): Batch size for dataloaders
        
    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader, tokenizer)
    """
    # Ensure processed data directory exists
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    
    # Load and preprocess data if not provided
    if data_df is None:
        df = load_and_preprocess_data(create_caller_only=True)
    else:
        df = data_df
    
    # Split data
    train_df, val_df, test_df = split_data(df)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
    
    # Prepare dataloaders
    train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(
        train_df, val_df, test_df, tokenizer, batch_size=batch_size, use_caller_only=use_caller_only
    )
    
    return train_dataloader, val_dataloader, test_dataloader, tokenizer


if __name__ == "__main__":
    # Test data preprocessing
    df = load_and_preprocess_data()
    train_df, val_df, test_df = split_data(df)
    
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
    train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(
        train_df, val_df, test_df, tokenizer
    )
    
    print("Data preprocessing test successful!") 