FROM python:3.10-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install only the necessary dependencies and clean up in a single layer
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader punkt && \
    rm -rf /root/.cache/pip

# Copy the source code
COPY src/ /app/src/
COPY new_api_server.py .

# Create directory structure for model
RUN mkdir -p /app/output/models_20250321_222124/

# Copy model files directly into the container
# This ensures the model works with plain 'docker run' without volume mounts
COPY outputs/models_20250321_222124/lstm_cnn_roberta_full.pt /app/output/models_20250321_222124/
COPY outputs/models_20250321_222124/lstm_cnn_roberta_full_vocab.json /app/output/models_20250321_222124/
COPY outputs/models_20250321_222124/lstm_cnn_roberta_full_metadata.json /app/output/models_20250321_222124/

# Set environment variables with EXACT paths matching those in new_api_server.py
# Note the missing leading slash in "output/" - this must match what the server expects
ENV MODEL_PATH=output/models_20250321_222124/lstm_cnn_roberta_full.pt
ENV MODEL_VOCAB_PATH=output/models_20250321_222124/lstm_cnn_roberta_full_vocab.json
ENV PYTHONPATH=/app:${PYTHONPATH}

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "new_api_server:app", "--host", "0.0.0.0", "--port", "8000"] 