# Complaint Detection API Service

This is a FastAPI service for detecting complaints in customer service conversations using an LSTM-CNN model with RoBERTa tokenization.

## Deployment Instructions

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available for the container

### Quick Start

1. Clone this repository
2. Create the model directory structure:
   ```bash
   mkdir -p outputs/models_20250321_222124
   ```
3. Place your model files in this directory:
   - `lstm_cnn_roberta_full.pt` (model weights)
   - `lstm_cnn_roberta_full_vocab.json` (vocabulary)
   - `lstm_cnn_roberta_full_metadata.json` (model metadata)

4. Run the deployment script:
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```
   
   Or use Docker Compose directly:
   ```bash
   docker-compose up -d
   ```

5. Access the API at http://localhost:8000/docs

### Configuration

You can configure the service by:

1. Editing the `.env` file (copy from `.env.example`)
2. Modifying environment variables in `docker-compose.yml`

### Manual Docker Build

If you prefer not to use Docker Compose:

```bash
# Build the Docker image
docker build -t complaint-api .

# Run the container
docker run -d -p 8000:8000 \
  -v ./outputs/models_20250321_222124:/app/output/models_20250321_222124:ro \
  --name complaint-detection-api \
  complaint-api
```

## API Usage

The API provides two main endpoints:

1. `/analyze/chunk` - Analyze a single conversation chunk
2. `/analyze/conversation` - Analyze a full conversation with sliding window

Authentication is required using a demo token:

```
Bearer demo_token
```

## Example Request

```bash
curl -X POST "http://localhost:8000/analyze/conversation" \
  -H "Authorization: Bearer demo_token" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Caller: I've been trying to resolve this issue for weeks.\nAgent: I understand your frustration. Let me help you.\nCaller: This is the third time I've called about this.\nAgent: I apologize for the inconvenience. Let's get this resolved.",
    "chunk_size": 4
  }'
```

## Troubleshooting

### Missing Model Files
If you encounter build errors related to missing model files, make sure:
1. The model directory structure exists: `outputs/models_20250321_222124/`
2. The model files are in place before running docker-compose
3. You have at least 4GB of available memory for the container

### Docker Compose Not Found
The deployment script supports both `docker-compose` and `docker compose` command formats.

## Model Information

- Type: LSTM-CNN with RoBERTa tokenization
- Accuracy: 80.9%
- Precision: 89.4%
- F1 Score: 0.775 