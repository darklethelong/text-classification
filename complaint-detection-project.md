# Customer Complaint Detection System: Project Plan & Workflow

## Project Overview

This project aims to build a system that can automatically detect complaints in call center conversations. The system will analyze conversations between agents and callers to identify when callers express complaints, allowing for better monitoring of customer satisfaction and potential service improvements.

### Data Structure
- **Input Data Format**: Conversations split into chunks of 4 consecutive utterances between Agent and Caller
- **Labels**: Binary classification (complaint vs. non-complaint)
- **Maximum Input Length**: 1024 tokens

## Project Workflow

### 1. Data Preparation and Preprocessing

1. **Data Collection**
   - Collect and digitize call center conversations
   - Ensure proper transcription with timestamps
   - Split conversations into chunks of 4 consecutive utterances

2. **Data Annotation**
   - Label each chunk as "complaint" or "non-complaint"
   - Consider intensity of complaints (mild, moderate, significant, high)
   - Calculate complaint percentage per conversation

3. **Data Preprocessing**
   - Clean text data (remove special characters, normalize text)
   - Create two versions of the dataset:
     - Full conversation chunks (Agent + Caller utterances)
     - Caller-only utterances
   - Tokenize text data
   - Split data into training (70%), validation (15%), and test (15%) sets

### 2. Model Development

We will develop and compare six different models for complaint detection:

#### Model 1: Full Conversation MLM Approach
- Use original chunks (Agent + Caller utterances)
- Fine-tune jinaai/jina-embeddings-v2-small-en with MLM objective
- Add classification head
- Train on binary classification task

#### Model 2: Caller-Only MLM Approach
- Use only Caller's utterances
- Fine-tune jinaai/jina-embeddings-v2-small-en with MLM objective
- Add classification head
- Train on binary classification task

#### Model 3: Full Conversation LSTM-CNN Approach
- Use original chunks (Agent + Caller utterances)
- Implement custom architecture with:
  - Embedding layer
  - LSTM layer(s)
  - CNN layer(s)
  - Dense layers for classification

#### Model 4: Caller-Only LSTM-CNN Approach
- Use only Caller's utterances
- Implement custom architecture with:
  - Embedding layer
  - LSTM layer(s)
  - CNN layer(s)
  - Dense layers for classification

#### Model 5: Full Conversation Hybrid Approach
- Use original chunks (Agent + Caller utterances)
- Fine-tune jinaai/jina-embeddings-v2-small-en with MLM objective
- Add custom LSTM and CNN layers
- Add dense layers for classification

#### Model 6: Caller-Only Hybrid Approach
- Use only Caller's utterances
- Fine-tune jinaai/jina-embeddings-v2-small-en with MLM objective
- Add custom LSTM and CNN layers
- Add dense layers for classification

### 3. Model Training and Evaluation

1. **Training Process**
   - Train all models with the same hyperparameters for fair comparison
   - Use early stopping and learning rate scheduling
   - Track training and validation loss/accuracy
   - Save model checkpoints

2. **Evaluation Metrics**
   - Primary metric: F1-score on test set
   - Additional metrics: Precision, Recall, Accuracy
   - Latency measurements (inference time)

### 4. Performance Comparison

1. **Model Comparison**
   - Compare all six models based on F1-score
   - Analyze precision-recall tradeoffs
   - Measure and compare inference latency
   - Create visualization charts for performance comparison

2. **Analysis Factors**
   - Impact of using full conversation vs. caller-only utterances
   - Effect of different model architectures (MLM, LSTM-CNN, Hybrid)
   - Tradeoff between accuracy and inference speed

### 5. Final Model Selection and Deployment

1. **Best Model Selection**
   - Select the model with the best balance of F1-score and latency
   - Perform final fine-tuning if necessary

2. **Deployment Pipeline**
   - Create a streamlined inference pipeline
   - Implement percentage calculation for complaints in conversations
   - Develop visualization dashboard with:
     - Timeline of complaint occurrences by timestamp
     - Complaint percentage chart
     - Intensity analysis of complaints

### 6. Visualization and Reporting

1. **Visualization Components**
   - Complaint timeline chart showing complaints by timestamp
   - Percentage gauge showing overall complaint level
   - Heatmap of complaint intensity throughout conversation

2. **Reporting Features**
   - Generate automated reports with complaint statistics
   - Highlight conversation segments with high complaint density
   - Provide actionable insights for customer service improvement

## Technical Implementation Details

### Data Processing Pipeline

```
Raw Conversation Data
    ↓
Transcription & Timestamps
    ↓
Chunking (4 consecutive utterances)
    ↓
Create dual datasets (Full & Caller-only)
    ↓
Tokenization & Feature Engineering
    ↓
Train-Val-Test Split
```

### Model Architecture Specifics

1. **MLM-based Models (1, 2)**
   - Base model: jinaai/jina-embeddings-v2-small-en
   - Fine-tuning approach: Masked Language Modeling
   - Classification head: Linear layer with softmax activation

2. **LSTM-CNN Models (3, 4)**
   - Embedding: Pre-trained word embeddings or learned embeddings
   - LSTM: Bidirectional LSTM with 128-256 units
   - CNN: Multiple kernels of different sizes (3, 5, 7)
   - Pooling: Global max pooling
   - Dense: Multiple dense layers with dropout

3. **Hybrid Models (5, 6)**
   - Base model: Fine-tuned jinaai/jina-embeddings-v2-small-en
   - Additional layers:
     - Bidirectional LSTM layer (128 units)
     - CNN layer with multiple kernel sizes
     - Global max pooling
     - Dense layers with dropout

### Inference Pipeline

```
Input Conversation
    ↓
Preprocessing (Chunking, Tokenization)
    ↓
Model Inference (Selected best model)
    ↓
Complaint Detection & Classification
    ↓
Percentage Calculation
    ↓
Visualization Generation
    ↓
Report Output
```

## Expected Outcomes

1. A robust complaint detection system with high F1-score
2. Comparative analysis of different model architectures
3. Insights into the importance of context (agent utterances) vs. caller-only utterances
4. A visualization system that helps track complaint patterns
5. A scalable solution that can be extended to other customer service metrics

## Future Enhancements

1. Multi-class classification for complaint types
2. Sentiment intensity scoring
3. Real-time complaint detection during live calls
4. Integration with customer satisfaction metrics
5. Root cause analysis of complaints
