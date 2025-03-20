# Customer Complaint Detection System

This project implements a system for automatically detecting complaints in call center conversations. The system analyzes conversations between agents and callers to identify when callers express complaints, allowing for better monitoring of customer satisfaction and potential service improvements.

## Features

- Processes conversation data into chunks of 4 consecutive utterances
- Supports both full conversation analysis and caller-only utterance analysis
- Implements and compares four model architectures:
  - LSTM-CNN on full conversations
  - LSTM-CNN on caller-only utterances
  - MLM-based approach (using jinaai/jina-embeddings-v2-small-en) on full conversations
  - MLM-based approach on caller-only utterances
- Provides comprehensive evaluation metrics and visualizations
- Generates complaint reports with percentage calculation and intensity analysis

## Project Structure

```
├── README.md
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── data
│   │   ├── __init__.py
│   │   └── preprocessor.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── lstm_cnn_model.py
│   │   └── mlm_model.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── training.py
│   │   └── inference.py
│   ├── evaluation
│   │   ├── __init__.py
│   │   └── visualizations.py
│   ├── train_and_evaluate.py
│   └── inference_demo.py
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd customer-complaint-detection
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

Place your conversation data in CSV format with columns 'text' and 'label' (where label is either 'complaint' or 'non-complaint').

### Training Models

Train all models:
```bash
python src/train_and_evaluate.py --data_file train.csv --output_dir output
```

Train specific models:
```bash
python src/train_and_evaluate.py --data_file train.csv --output_dir output --models lstm_cnn_full
```

Available model options: 
- `all` (default)
- `lstm_cnn_full`
- `lstm_cnn_caller_only`
- `mlm_full`
- `mlm_caller_only`

### Running Inference

After training, you can use the models for inference:

```bash
python src/inference_demo.py --model_path output/models_YYYYMMDD_HHMMSS/lstm_cnn_full.pt --model_type lstm-cnn
```

You can either provide a conversation file:
```bash
python src/inference_demo.py --model_path output/models_YYYYMMDD_HHMMSS/lstm_cnn_full.pt --model_type lstm-cnn --input_file conversation.txt
```

Or enter a conversation interactively when prompted.

## Evaluation Metrics

The system evaluates models using:
- Accuracy
- Precision
- Recall
- F1-score
- Inference latency

The results are saved as JSON metadata files, CSV comparison tables, and visualizations.

## License

[Specify your license here] 