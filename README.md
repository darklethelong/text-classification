# Customer Complaint Detection System

A comprehensive system for detecting and analyzing customer complaints in call center conversations. This project implements multiple machine learning models to classify conversation segments as either containing a complaint or not.

## Features

- **Multiple Model Architectures**: 
  - MLM-based models (using pretrained transformer)
  - LSTM-CNN models (custom architecture)
  - Hybrid models (combining transformer + LSTM/CNN)
  - Each architecture trained on full conversations or caller-only utterances

- **Complete Processing Pipeline**:
  - Text preprocessing (cleaning, normalization)
  - Caller extraction from conversations
  - Training and evaluation pipelines
  - Inference system for making predictions

- **Visualization & Reporting**:
  - Complaint timeline visualization
  - Percentage gauges and distribution charts
  - Intensity heatmaps
  - Word clouds for common complaint terms
  - Interactive HTML reports

## Project Structure

```
customer-complaint-detection/
├── config.py                  # Configuration parameters
├── main.py                    # Main entry point
├── model_comparison.py        # Script for comparing models
├── data/                      # Data directory
│   ├── sample_data.csv        # Sample dataset
│   └── synthetic_call_center_data.csv  # Synthetic dataset
├── models/                    # Model implementations
│   ├── model_factory.py       # Factory for model creation
│   ├── mlm_models.py          # MLM-based models
│   ├── lstm_cnn_models.py     # LSTM-CNN models
│   └── hybrid_models.py       # Hybrid models
├── utils/                     # Utility modules
│   ├── data_preprocessing.py  # Data preprocessing
│   ├── training.py            # Training functions
│   └── evaluation.py          # Evaluation functions
├── inference/                 # Inference pipeline
│   └── inference.py           # Inference functionality
└── visualization/             # Visualization tools
    └── visualization.py       # Visualization functions
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/darklethelong/text-classification.git
cd text-classification
```

2. Create a virtual environment and activate it:
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Data Format

The system expects a CSV file with the following structure:
- `text` or `full_text`: Column containing the conversation text
- `label`: Binary column (0 for non-complaint, 1 for complaint)
- `timestamp` (optional): For timeline visualizations

Examples are available in the `data/` directory.

## Quick Start

### Training Models

Train a single model:

```bash
python model_comparison.py --data_path data/synthetic_call_center_data.csv --model_types mlm_full --num_epochs 3 
```

Train and compare multiple models:

```bash
python model_comparison.py --data_path data/synthetic_call_center_data.csv --model_types mlm_full lstm_cnn_full --num_epochs 3 --generate_report
```

### Using main.py Interface

Train a model:

```bash
python main.py train --model_type mlm_full --data_path data/synthetic_call_center_data.csv --num_epochs 3
```

Evaluate models:

```bash
python main.py evaluate --evaluate_all 
```

Make predictions on new data:

```bash
python main.py predict --input_file data/synthetic_call_center_data.csv --model_type mlm_full
```

Generate visualizations:

```bash
python main.py visualize --input_file path/to/predictions.csv --generate_report
```

## Implementation Instructions

### 1. Setting Up a New Project

1. Clone the repository and install dependencies as shown above.
2. Prepare your data following the format in `data/sample_data.csv`.
3. Adjust hyperparameters in `config.py` as needed.

### 2. Training Your Own Models

1. Start with a simple model like `mlm_full` to establish a baseline:
   ```bash
   python model_comparison.py --data_path your_data.csv --model_types mlm_full --num_epochs 3
   ```

2. Compare multiple model architectures:
   ```bash
   python model_comparison.py --data_path your_data.csv --model_types mlm_full lstm_cnn_full hybrid_full --num_epochs 3 --generate_report
   ```

3. Tune hyperparameters:
   - Adjust batch size with `--batch_size`
   - Modify learning rate and other parameters in `config.py`

### 3. Making Predictions

1. Determine the best model from your training/evaluation:
   ```bash
   python main.py evaluate --evaluate_all
   ```

2. Use the best model for predictions:
   ```bash
   python main.py predict --input_file new_data.csv --model_type best_model_type
   ```

3. Generate a report with visualizations:
   ```bash
   python main.py visualize --input_file predictions.csv --generate_report
   ```

### 4. Extending the System

To add new model architectures:
1. Create a new model file (e.g., `models/your_model.py`)
2. Update `models/model_factory.py` to include your model
3. Implement the required forward pass and model creation functions

## Examples

Comparing two models with report generation:

```bash
python model_comparison.py --data_path data/synthetic_call_center_data.csv --model_types mlm_full lstm_cnn_full --num_epochs 3 --generate_report
```

Training a single model with caller-only data:

```bash
python main.py train --model_type lstm_cnn_caller_only --data_path data/synthetic_call_center_data.csv --num_epochs 5
```

## Troubleshooting

- **"CUDA out of memory" error**: Reduce batch size with `--batch_size`
- **"File not found" errors**: Check file paths and create necessary directories
- **Poor model performance**: Increase training epochs, adjust learning rate, or try a different model architecture

## License

This project is available under the MIT License.

## Acknowledgements

This project utilizes the Hugging Face Transformers library and PyTorch. 