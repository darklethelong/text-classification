#!/bin/bash
# Simple deployment script for Complaint Detection API

# Make the script executable
# chmod +x deploy.sh

# Display banner
echo "================================================"
echo "Complaint Detection API Deployment"
echo "================================================"

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check Docker Compose installation
# First try docker-compose
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
# If not available, try docker compose (new style)
elif docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "Using Docker Compose command: $DOCKER_COMPOSE"

# Check if model files exist
echo "Checking for model files..."
MODEL_DIR="outputs/models_20250321_222124"
MODEL_FILE="$MODEL_DIR/lstm_cnn_roberta_full.pt"
VOCAB_FILE="$MODEL_DIR/lstm_cnn_roberta_full_vocab.json"
METADATA_FILE="$MODEL_DIR/lstm_cnn_roberta_full_metadata.json"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found at $MODEL_DIR"
    echo "Creating directory structure..."
    mkdir -p "$MODEL_DIR"
fi

# Ask user to confirm if files are missing but proceeding anyway
if [ ! -f "$MODEL_FILE" ] || [ ! -f "$VOCAB_FILE" ] || [ ! -f "$METADATA_FILE" ]; then
    echo "Warning: One or more model files are missing:"
    [ ! -f "$MODEL_FILE" ] && echo "- Missing model file: $MODEL_FILE"
    [ ! -f "$VOCAB_FILE" ] && echo "- Missing vocabulary file: $VOCAB_FILE"
    [ ! -f "$METADATA_FILE" ] && echo "- Missing metadata file: $METADATA_FILE"
    
    read -p "Do you want to proceed anyway? (y/N): " proceed
    if [[ ! "$proceed" =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled."
        exit 1
    fi
    
    echo "Proceeding with deployment, but service may not work correctly without model files."
    echo "Make sure to mount the correct model files as volumes."
fi

# Build and start containers
echo "Building and starting Docker containers..."
$DOCKER_COMPOSE down
$DOCKER_COMPOSE build
$DOCKER_COMPOSE up -d

# Check if container is running
if [ "$($DOCKER_COMPOSE ps -q | wc -l)" -gt 0 ]; then
    echo "Deployment successful!"
    echo "API is now running at http://localhost:8000"
    echo "API documentation: http://localhost:8000/docs"
else
    echo "Error: Deployment failed. Check logs with '$DOCKER_COMPOSE logs'."
    exit 1
fi 