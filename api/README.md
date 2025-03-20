# Complaint Detection API

A FastAPI-based API service for detecting complaints in customer service conversations using a trained LSTM-CNN model.

## Features

- Token-based authentication using JWT
- API endpoints for complaint detection
- Health check endpoint
- Swagger documentation
- Professional structure with modular components
- Stress testing support

## Project Structure

```
api/
├── config.py               # Configuration settings
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Dependencies
├── stress_test.py          # Locust stress test script
├── routers/                # API route handlers
│   ├── __init__.py
│   ├── auth.py             # Authentication routes
│   └── complaints.py       # Complaint detection routes
├── schemas/                # Pydantic data models
│   ├── __init__.py
│   ├── auth.py             # Authentication schemas
│   └── conversations.py    # Conversation schemas
└── services/               # Business logic services
    ├── __init__.py
    ├── auth.py             # Authentication service
    └── model_service.py    # Model service for predictions
```

## Installation

1. Clone the repository
2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install the dependencies:
```bash
pip install -r api/requirements.txt
```

## Configuration

The API can be configured using environment variables or a `.env` file. Available settings:

- `APP_NAME`: Name of the application
- `DEBUG`: Enable debug mode (True/False)
- `API_PREFIX`: Prefix for API endpoints
- `SECRET_KEY`: Secret key for JWT token generation
- `ALGORITHM`: Algorithm for JWT token generation
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Token expiration time in minutes
- `MODEL_PATH`: Path to the trained model file
- `ALLOW_ORIGINS`: Comma-separated list of allowed origins for CORS

Example `.env` file:
```
DEBUG=True
SECRET_KEY=your_secret_key_here
MODEL_PATH=output/models_20250320_124018/lstm_cnn_full.pt
```

## Running the API

Start the API server with:

```bash
cd api
python main.py
```

The API server will be available at `http://localhost:8000`.

Access the interactive API documentation at `http://localhost:8000/docs`.

## Authentication

The API uses JWT token authentication. To authenticate:

1. Make a POST request to `/token` with form data containing `username` and `password`
2. Use the returned access token in the `Authorization` header for subsequent requests:
   ```
   Authorization: Bearer <access_token>
   ```

## API Endpoints

### Authentication

- `POST /token`: Get an access token
- `GET /users/me`: Get information about the currently authenticated user

### Complaint Detection

- `POST /api/predict`: Analyze a conversation for complaints
- `GET /api/health`: Check the health of the API

## Example Usage

### Authentication

```python
import requests

# Get token
response = requests.post(
    "http://localhost:8000/token",
    data={"username": "admin", "password": "adminpassword"},
    headers={"Content-Type": "application/x-www-form-urlencoded"}
)

token = response.json()["access_token"]
```

### Making a Prediction

```python
# Example conversation
conversation = """
Caller: Hello, I've been trying to get my internet fixed for a week now and nobody seems to care.
Agent: I'm sorry to hear that. Let me look into this for you.
Caller: I've called three times already and each time I was promised someone would come, but nobody ever showed up.
"""

# Make prediction
response = requests.post(
    "http://localhost:8000/api/predict",
    json={"text": conversation},
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
)

result = response.json()
print(f"Is complaint: {result['is_complaint']}")
print(f"Confidence: {result['confidence']}")
```

## Stress Testing

The API includes a Locust-based stress test script. To run the stress test:

1. Install Locust:
```bash
pip install locust
```

2. Run the stress test:
```bash
cd api
locust -f stress_test.py
```

3. Open the Locust web interface at http://localhost:8089 and configure your test parameters

## Production Deployment

For production deployment, consider:

1. Using a proper database for user management (e.g., PostgreSQL)
2. Setting up HTTPS with a proper TLS certificate
3. Implementing rate limiting
4. Using a process manager like Gunicorn or Uvicorn with multiple workers
5. Setting up proper logging
6. Using a reverse proxy like Nginx

Example deployment command with Gunicorn:
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
``` 