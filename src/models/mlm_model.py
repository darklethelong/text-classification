import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MLMClassifier(nn.Module):
    def __init__(self, model_name="/app/mlm_model", num_classes=2, dropout=0.1):
        super(MLMClassifier, self).__init__()
        
        # Load pre-trained transformer model
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use CLS token representation (or pooler output) for classification
        cls_output = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token representation
        
        # Apply dropout for regularization
        cls_output = self.dropout(cls_output)
        
        # Get logits
        logits = self.classifier(cls_output)
        
        return logits

def get_mlm_model_and_tokenizer(model_name="/app/mlm_model", num_classes=2, dropout=0.1):
    """Helper function to get both the model and tokenizer."""
    model = MLMClassifier(model_name, num_classes, dropout)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer 