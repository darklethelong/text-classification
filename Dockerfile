# Multi-stage build for Complaint Detection System

# Stage 1: Build the React UI
FROM node:16-alpine as ui-builder
WORKDIR /app/ui
COPY ui/package*.json ./
RUN npm install
COPY ui/ ./
RUN npm run build

# Stage 2: Python API with UI serving
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY *.py ./
COPY ./models/ ./models/
COPY ./src/ ./src/

# Copy built UI from the UI builder stage
COPY --from=ui-builder /app/ui/build/ ./static/

# Add environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV HOST=0.0.0.0
ENV MODEL_PATH=/app/models
ENV TOKEN_SECRET_KEY=CHANGE_ME_IN_PRODUCTION

# Expose the port the app runs on
EXPOSE 8000

# Start command
CMD ["python", "new_api_server.py"]

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1 