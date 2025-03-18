# Customer Complaint Detection API

This API provides real-time complaint detection with visualization capabilities using FastAPI. It serves as an interface to the trained machine learning models for detecting complaints in customer conversations.

## Features

- Single text complaint detection
- Batch processing of multiple texts
- Real-time analysis of conversations with chunking (process after every N messages)
- WebSocket support for continuous streaming
- Interactive visualization of complaints
- Session management for tracking conversations
- Caching mechanism for visualizations

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- FastAPI
- Uvicorn
- Other dependencies in `requirements.txt`

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure you have trained models in the `models/` directory. The default model path is `models/mlm_full.pt`.

## Running the API

To run the API, use the `run_api.py` script:

```bash
python run_api.py
```

This will start the server on http://localhost:8000. You can access the API documentation at http://localhost:8000/docs.

## API Endpoints

### Single Text Analysis

**Endpoint**: `POST /predict`

**Request Body**:
```json
{
  "text": "Your customer text here",
  "session_id": null
}
```

**Response**:
```json
{
  "is_complaint": true,
  "complaint_probability": 0.85,
  "text": "Your customer text here",
  "timestamp": "2023-03-18T15:30:45",
  "complaint_intensity": "High"
}
```

### Batch Processing

**Endpoint**: `POST /predict/batch`

**Request Body**:
```json
{
  "texts": [
    "First customer text",
    "Second customer text",
    "..."
  ],
  "session_id": null
}
```

**Query Parameters**:
- `generate_viz` (boolean): Whether to generate visualizations

**Response**:
```json
{
  "results": [
    {
      "is_complaint": true,
      "complaint_probability": 0.85,
      "text": "First customer text",
      "timestamp": "2023-03-18T15:30:45",
      "complaint_intensity": "High"
    },
    ...
  ],
  "complaint_percentage": 0.5,
  "visualizations": {
    "timeline": "base64_encoded_image",
    "gauge": "base64_encoded_image",
    "heatmap": "base64_encoded_image",
    "distribution": "base64_encoded_image"
  }
}
```

### Create Session

**Endpoint**: `POST /sessions/create`

**Request Body**:
```json
{
  "session_name": "Customer Support Call #12345",
  "threshold": 0.3,
  "chunk_size": 4
}
```

**Response**:
```json
{
  "session_id": "session_1679154645",
  "name": "Customer Support Call #12345",
  "created_at": "2023-03-18T15:30:45",
  "message_count": 0,
  "complaint_count": 0,
  "complaint_percentage": 0.0
}
```

### Add Message to Session

**Endpoint**: `POST /sessions/{session_id}/add_message`

**Path Parameters**:
- `session_id`: ID of the session

**Request Body**:
```json
{
  "text": "Your customer text here"
}
```

**Response** (when chunk is not complete):
```json
{
  "processed": false,
  "chunk_size": 4,
  "messages_in_chunk": 1,
  "messages_needed": 3
}
```

**Response** (when chunk is complete):
```json
{
  "processed": true,
  "chunk_size": 4,
  "results": [
    {
      "is_complaint": true,
      "complaint_probability": 0.85,
      "text": "Your customer text here",
      "timestamp": "2023-03-18T15:30:45",
      "complaint_intensity": "High"
    },
    ...
  ]
}
```

### List Sessions

**Endpoint**: `GET /sessions`

**Response**:
```json
[
  {
    "session_id": "session_1679154645",
    "name": "Customer Support Call #12345",
    "created_at": "2023-03-18T15:30:45",
    "message_count": 10,
    "complaint_count": 3,
    "complaint_percentage": 0.3
  },
  ...
]
```

### Get Session Data

**Endpoint**: `GET /sessions/{session_id}/data`

**Path Parameters**:
- `session_id`: ID of the session

**Response**:
```json
{
  "session_info": {
    "session_id": "session_1679154645",
    "name": "Customer Support Call #12345",
    "created_at": "2023-03-18T15:30:45",
    "message_count": 10,
    "complaint_count": 3,
    "complaint_percentage": 0.3
  },
  "data": [
    {
      "text": "Your customer text here",
      "is_complaint": true,
      "complaint_probability": 0.85,
      "timestamp": "2023-03-18T15:30:45",
      "complaint_intensity": "High"
    },
    ...
  ]
}
```

### Get Session Visualizations

**Endpoint**: `GET /sessions/{session_id}/visualizations`

**Path Parameters**:
- `session_id`: ID of the session

**Response**:
```json
{
  "visualizations": {
    "timeline": "base64_encoded_image",
    "gauge": "base64_encoded_image",
    "heatmap": "base64_encoded_image",
    "distribution": "base64_encoded_image"
  }
}
```

### Delete Session

**Endpoint**: `DELETE /sessions/{session_id}`

**Path Parameters**:
- `session_id`: ID of the session

**Response**:
```json
{
  "message": "Session session_1679154645 deleted"
}
```

## WebSocket Support

For real-time analysis, you can use the WebSocket endpoint:

**Endpoint**: `ws://localhost:8000/ws/{session_id}`

**Path Parameters**:
- `session_id`: ID of the session

**Messages to send**:
```json
{
  "text": "Your customer text here"
}
```

**Messages received**:

1. Connection established:
```json
{
  "type": "connection_established",
  "session_id": "session_1679154645",
  "chunk_size": 4
}
```

2. Message received (not enough for processing):
```json
{
  "type": "message_received",
  "messages_in_buffer": 1,
  "messages_needed": 3
}
```

3. Chunk processed:
```json
{
  "type": "chunk_processed",
  "results": [
    {
      "is_complaint": true,
      "complaint_probability": 0.85,
      "text": "Your customer text here",
      "timestamp": "2023-03-18T15:30:45",
      "complaint_intensity": "High"
    },
    ...
  ],
  "visualizations": {
    "timeline": "base64_encoded_image",
    "gauge": "base64_encoded_image",
    "heatmap": "base64_encoded_image",
    "distribution": "base64_encoded_image"
  },
  "session_info": {
    "message_count": 10,
    "complaint_count": 3,
    "complaint_percentage": 0.3
  }
}
```

## Web Client

A simple web client is included at `http://localhost:8000/` with three tabs:

1. **Single Text**: Analyze a single text for complaints
2. **Batch Processing**: Analyze multiple texts at once with visualizations
3. **Real-time Analysis**: Create a session and analyze texts in real-time using WebSockets

## Visualization Types

The API provides several visualization types:

- **Timeline**: Shows complaint probabilities over time
- **Gauge**: Shows the overall complaint percentage
- **Heatmap**: Shows the intensity of complaints
- **Distribution**: Shows the distribution of complaint probabilities

## Error Handling

All endpoints include proper error handling. If an error occurs, the response will include a status code and error message:

```json
{
  "detail": "Error message here"
}
```

## Performance Considerations

- The API uses a singleton pattern for the model to avoid loading it multiple times
- Visualizations are cached to improve performance
- Background tasks are used for time-consuming operations

## Security Considerations

- Add authentication in production
- Set proper CORS settings for production
- Limit request rates to prevent abuse

## Future Improvements

- Add authentication
- Add model retraining endpoint
- Add feedback mechanism for improving the model
- Add more visualization types
- Add support for more languages 