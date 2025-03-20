# Model Comparison Results

This document summarizes the performance of different models for complaint detection. All models were trained with 50 sample rows and 2 epochs.

## Model Performance Comparison

| Model               | Accuracy | Precision | Recall | F1 Score | Avg Latency (ms) |
|--------------------|----------|-----------|--------|----------|-----------------|
| LSTM-CNN (Full)     | 0.6      | 1.0       | 0.333  | 0.50     | 620.82          |
| LSTM-CNN (Caller-Only) | 0.6   | 0.6       | 1.0    | 0.75     | 573.95          |
| MLM (Full)          | 0.4      | 0.0       | 0.0    | 0.0      | 3772.87         |
| MLM (Caller-Only)   | 0.6      | 0.6       | 1.0    | 0.75     | 3588.02         |

## Sample Conversation Testing

For the purpose of testing, we used two sample conversations:
1. A complaint conversation (Customer with internet service issues)
2. A neutral conversation (Customer upgrading their plan)

### Model Behavior Summary

**LSTM-CNN (Full)**
- Tends to predict "no complaint" for both sample conversations
- Probability scores hover around 0.48-0.49, just below the threshold of 0.5
- Fast inference time (average 330-360ms)

**LSTM-CNN (Caller-Only)**
- Detects complaints in both conversations
- Probability scores are just above the threshold (0.503-0.506)
- Fast inference time (average 330-390ms)
- Does not differentiate well between complaint and non-complaint conversations

**MLM (Caller-Only)**
- Detects complaints in both conversations
- Shows good probability separation: 
  - For complaint conversation: 0.53-0.68
  - For non-complaint conversation: 0.64-0.73
- Higher inference latency (approximately 900ms)
- Has the strongest confidence in non-complaint conversation being a complaint (false positive)

## Observations and Recommendations

1. **LSTM-CNN (Full)** properly loads the vocabulary but has classification issues, consistently predicting "no complaint" across both sample conversations. This suggests:
   - The model might be overfitted to the limited training data (50 samples)
   - Classification threshold might need adjustment (probabilities are close to 0.5)
   - More training epochs may be needed to reach optimal performance

2. **LSTM-CNN (Caller-Only)** classifies everything as a complaint with low confidence (just above threshold). This indicates:
   - Focusing only on caller utterances might over-simplify the task
   - More training data is required for better discrimination

3. **MLM (Caller-Only)** has the highest probability scores but shows incorrect classification on the non-complaint conversation. Despite having the best F1 score in validation (0.75), it:
   - Has the highest latency, which may be problematic for real-time applications
   - Requires significant computational resources
   - Needs better calibration for threshold setting

## Next Steps

1. **Increase training data**: Train models with at least 500 samples to reduce overfitting.
2. **More training epochs**: Increase epochs to 5-10 to allow models to converge properly.
3. **Threshold tuning**: Adjust classification thresholds based on ROC curves.
4. **Balanced dataset**: Ensure balanced representation of complaint and non-complaint conversations.
5. **Model ensemble**: Consider combining predictions from multiple models for better accuracy.
6. **Feature engineering**: Add additional features like sentiment scores to improve discrimination.

By addressing these areas, we can significantly improve model performance and reliability for complaint detection. 