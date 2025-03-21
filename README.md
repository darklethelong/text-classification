# Complaint Detection System

A machine learning-based system for detecting complaints in customer service conversations. The system analyzes conversation chunks to identify complaints and provides visualization of complaint trends.

## Features

- Detects complaints in customer service conversations using LSTM-CNN with RoBERTa tokenizer
- Processes conversations in configurable chunks for detailed analysis
- Provides visual complaint tracking with multiple visualization options
- Includes token-based authentication for API access
- Features a cyberpunk-themed UI with Matrix falling code background
- Fully containerized for easy deployment

## Components

The system consists of two main components:

1. **FastAPI Backend** - A Python-based API server that:
   - Loads and utilizes ML models for complaint detection
   - Processes and analyzes conversation chunks
   - Provides JSON API for frontend integration
   - Secures access with token-based authentication

2. **React Frontend** - A modern UI built with React that:
   - Visualizes complaint detection results
   - Provides real-time complaint trend charts
   - Displays a cyberpunk-themed interface with Matrix rain effects
   - Allows for testing with sample conversations or custom input

## Deployment

See [DEPLOY.md](DEPLOY.md) for detailed deployment instructions.

Quick deployment with Docker:

```bash
# Make the deployment script executable
chmod +x deploy.sh

# Run the deployment script
./deploy.sh
```

## Repository Structure

- `/models` - Directory for ML model files
- `/data` - Directory for training and testing data
- `/src` - Source code for ML models and core functionality
- `/ui` - React frontend application
- `new_api_server.py` - Main API server application
- `Dockerfile` & `docker-compose.yml` - Docker configuration for deployment
- `deploy.sh` - Deployment script

## License

[MIT License](LICENSE) 