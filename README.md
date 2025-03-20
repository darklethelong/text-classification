# Customer Complaint Detection System

This project implements a system for automatically detecting complaints in call center conversations. The system analyzes conversations between agents and callers to identify when callers express complaints, allowing for better monitoring of customer satisfaction and potential service improvements.

## Features

- Processes conversation data into chunks of 4 consecutive utterances
- Supports both full conversation analysis and caller-only utterance analysis
- Implements and compares six model architectures:
  - LSTM-CNN with custom tokenizer on full conversations
  - LSTM-CNN with custom tokenizer on caller-only utterances
  - LSTM-CNN with RoBERTa tokenizer on full conversations
  - LSTM-CNN with RoBERTa tokenizer on caller-only utterances
  - MLM-based approach (using jinaai/jina-embeddings-v2-small-en) on full conversations
  - MLM-based approach on caller-only utterances
- Provides comprehensive evaluation metrics and visualizations
- Generates complaint reports with percentage calculation and intensity analysis
- RESTful API server with token-based authentication for model inference
- User-friendly web UI for testing and visualizing complaint detection results

## Project Structure

```
├── README.md
├── requirements.txt
├── train.csv                       # Training data
├── main.py                         # Main script for running the system
├── inference.py                    # Inference script for using trained models
├── simple_api_server.py            # Standalone API server script
├── src/                            # Core code modules
│   ├── data/                       # Data preprocessing
│   ├── models/                     # Model definitions
│   ├── utils/                      # Utility functions
│   ├── evaluation/                 # Evaluation metrics and visualizations
│   └── train_and_evaluate.py       # Training and evaluation script
├── api/                            # API server components
│   ├── config.py                   # API configuration
│   ├── main.py                     # API entry point
│   ├── services/                   # Business logic
│   ├── routers/                    # API endpoints
│   └── schemas/                    # Data validation
└── ui/                             # React-based web UI
    ├── src/                        # UI source code
    ├── public/                     # Static assets
    └── package.json                # UI dependencies
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

3. For the UI:
```bash
cd ui
npm install
```

## Usage

### Data Preparation

Place your conversation data in CSV format with columns 'text' and 'label' (where label is either 'complaint' or 'non-complaint').

### Training Models

Train all models:
```bash
python src/train_and_evaluate.py --data_file train.csv --output_dir output
```

Train specific models with limited data and epochs:
```bash
python src/train_and_evaluate.py --data_file train.csv --output_dir output --models lstm_cnn_roberta_full --max_rows 100 --max_epochs 2
```

Available model options: 
- `all` (default)
- `lstm_cnn_custom_full`
- `lstm_cnn_custom_caller_only`
- `lstm_cnn_roberta_full`
- `lstm_cnn_roberta_caller_only`
- `mlm_full`
- `mlm_caller_only`

### Running the API Server

Start the standalone API server:
```bash
python simple_api_server.py
```

The API will be available at:
- API base URL: http://127.0.0.1:8000
- Documentation: http://127.0.0.1:8000/docs
- Main endpoint: POST /analyze

### Running the UI

1. Configure the UI environment:
```bash
cd ui
echo "REACT_APP_API_URL=http://localhost:8000" > .env
```

2. Start the UI:
```bash
npm start
```

3. Access the UI at http://localhost:3000

### Running Inference

Use the trained models for standalone inference:

```bash
python inference.py --model_path output/models_YYYYMMDD_HHMMSS/lstm_cnn_roberta_full.pt --model_type lstm-cnn
```

## Evaluation Metrics

The system evaluates models using:
- Accuracy
- Precision
- Recall
- F1-score
- Inference latency (CPU and GPU)

The results are saved as JSON metadata files, CSV comparison tables, and visualizations.

## License

[Specify your license here] 