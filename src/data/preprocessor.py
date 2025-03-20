import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np

# Download nltk resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=1024, caller_only=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.caller_only = caller_only
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # If caller_only, extract only the caller's utterances
        if self.caller_only:
            lines = text.split('\n')
            caller_lines = [line for line in lines if line.startswith('Caller:')]
            text = ' '.join(caller_lines)
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Convert inputs and labels to tensors
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }

class VocabBuilder:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = Counter()
        
    def build_vocab(self, texts):
        # Tokenize texts and count word frequencies
        for text in texts:
            tokens = word_tokenize(text.lower())
            self.word_freq.update(tokens)
        
        # Add words that meet minimum frequency to vocabulary
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        return self
    
    def __len__(self):
        return len(self.word2idx)

class LSTMCNNDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=1024, caller_only=False):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.caller_only = caller_only
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # If caller_only, extract only the caller's utterances
        if self.caller_only:
            lines = text.split('\n')
            caller_lines = [line for line in lines if line.startswith('Caller:')]
            text = ' '.join(caller_lines)
        
        # Tokenize and convert to indices
        tokens = word_tokenize(text.lower())
        indices = [self.vocab.word2idx.get(token, 1) for token in tokens]  # 1 is <UNK>
        
        # Pad or truncate sequence
        if len(indices) < self.max_length:
            indices += [0] * (self.max_length - len(indices))  # 0 is <PAD>
        else:
            indices = indices[:self.max_length]
        
        # Convert to tensors
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'indices': indices_tensor,
            'label': label_tensor
        }

def clean_text(text):
    """Clean and normalize text data."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, keeping essential punctuation
    text = re.sub(r'[^\w\s\.\,\?\!]', '', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def load_and_preprocess_data(file_path, test_size=0.15, val_size=0.15, random_state=42):
    """Load data, clean, and split into train, validation, and test sets."""
    # Load CSV file
    df = pd.read_csv(file_path)
    
    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Convert labels to binary values (0: non-complaint, 1: complaint)
    df['binary_label'] = df['label'].apply(lambda x: 1 if x == 'complaint' else 0)
    
    # First split: separate out test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['binary_label']
    )
    
    # Second split: separate train and validation from the remaining data
    val_adjusted_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_adjusted_size, random_state=random_state, 
        stratify=train_val_df['binary_label']
    )
    
    return {
        'train': {
            'texts': train_df['cleaned_text'].tolist(),
            'labels': train_df['binary_label'].tolist()
        },
        'val': {
            'texts': val_df['cleaned_text'].tolist(),
            'labels': val_df['binary_label'].tolist()
        },
        'test': {
            'texts': test_df['cleaned_text'].tolist(),
            'labels': test_df['binary_label'].tolist()
        },
        'full_df': df
    }

def prepare_dataloaders(data_dict, tokenizer=None, vocab=None, batch_size=32, max_length=1024, 
                        caller_only=False, model_type='mlm'):
    """Prepare DataLoader objects for training and evaluation."""
    
    if model_type.lower() == 'mlm':
        # For MLM-based models using Hugging Face transformers
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for MLM models.")
        
        train_dataset = ComplaintDataset(
            data_dict['train']['texts'], 
            data_dict['train']['labels'], 
            tokenizer, 
            max_length, 
            caller_only
        )
        
        val_dataset = ComplaintDataset(
            data_dict['val']['texts'], 
            data_dict['val']['labels'], 
            tokenizer, 
            max_length, 
            caller_only
        )
        
        test_dataset = ComplaintDataset(
            data_dict['test']['texts'], 
            data_dict['test']['labels'], 
            tokenizer, 
            max_length, 
            caller_only
        )
    
    elif model_type.lower() == 'lstm-cnn':
        # For LSTM-CNN models
        if vocab is None:
            # Build vocabulary if not provided
            vocab_builder = VocabBuilder()
            vocab_builder.build_vocab(data_dict['train']['texts'])
            vocab = vocab_builder
        
        train_dataset = LSTMCNNDataset(
            data_dict['train']['texts'], 
            data_dict['train']['labels'], 
            vocab, 
            max_length, 
            caller_only
        )
        
        val_dataset = LSTMCNNDataset(
            data_dict['val']['texts'], 
            data_dict['val']['labels'], 
            vocab, 
            max_length, 
            caller_only
        )
        
        test_dataset = LSTMCNNDataset(
            data_dict['test']['texts'], 
            data_dict['test']['labels'], 
            vocab, 
            max_length, 
            caller_only
        )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'vocab': vocab
    }

def get_class_weights(labels):
    """Calculate class weights for imbalanced datasets."""
    class_counts = np.bincount(labels)
    total = len(labels)
    class_weights = torch.FloatTensor([total / (len(class_counts) * count) for count in class_counts])
    return class_weights 