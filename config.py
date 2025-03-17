"""
Configuration file for the complaint detection project.
Contains parameters for data processing, model architectures, training, and evaluation.
"""

import os
from datetime import datetime

# Data parameters
DATA_PATH = os.path.join("data", "sample_data.csv")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
MAX_LENGTH = 1024  # Maximum sequence length

# Text preprocessing parameters
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
RANDOM_SEED = 42
LOWER_CASE = True
REMOVE_SPECIAL_CHARS = True

# Model parameters
BASE_MODEL_NAME = "jinaai/jina-embeddings-v2-small-en"
OUTPUT_DIR = os.path.join("models", f"runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
MODELS_TO_TRAIN = ["mlm_full", "mlm_caller_only", "lstm_cnn_full", 
                  "lstm_cnn_caller_only", "hybrid_full", "hybrid_caller_only"]

# MLM model parameters
MLM_HIDDEN_SIZE = 768
MLM_DROPOUT = 0.1

# LSTM-CNN model parameters
LSTM_EMBEDDING_DIM = 300
LSTM_HIDDEN_DIM = 256
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_BIDIRECTIONAL = True
CNN_FILTERS = 128
CNN_KERNEL_SIZES = [3, 5, 7]
CNN_DROPOUT = 0.2

# Hybrid model parameters
HYBRID_LSTM_UNITS = 128
HYBRID_CNN_FILTERS = 64
HYBRID_DROPOUT = 0.2

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
EARLY_STOPPING_PATIENCE = 3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
CLASS_WEIGHTS = [0.22, 0.78]  # Based on 5k/6.4k non-complaint and 1.4k/6.4k complaint
SAVE_STEPS = 500
LOGGING_STEPS = 100

# Evaluation parameters
METRICS = ["accuracy", "precision", "recall", "f1"]
PRIMARY_METRIC = "f1"

# Visualization parameters
VISUALIZE_TRAINING = True
CONFUSION_MATRIX = True
ROC_CURVE = True
PRECISION_RECALL_CURVE = True
FEATURE_IMPORTANCE = True

# Inference parameters
COMPLAINT_THRESHOLD = 0.5
SAVE_PREDICTIONS = True
INFERENCE_BATCH_SIZE = 32 