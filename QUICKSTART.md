# Quick Start Guide

This guide will help you get up and running with the Customer Complaint Detection System quickly.

## Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-capable GPU (recommended for training)

## Installation

1. Install the required dependencies:
```
pip install -r requirements.txt
```

2. Ensure your data is in the correct format. The sample data is provided in `data/sample_data.csv`.

## Training Your First Model

For a quick test run, use the sample data:

```bash
# Train a simple MLM model with the sample data
python main.py train --model_type mlm_full --data_path data/sample_data.csv --output_dir models/ --num_epochs 3
```

This will train a model for a few epochs just to test the pipeline. For production use, you'll want to use more data and more epochs.

## Making Predictions

Once you have a trained model, you can use it to make predictions:

```bash
# Make predictions on new data
python main.py predict --input_file data/sample_data.csv --model_type mlm_full --model_path models/mlm_full.pt
```

This will output predictions to a CSV file in your output directory.

## Generating Visualizations

To visualize the results of your predictions:

```bash
# Generate visualizations from the prediction results
python main.py visualize --input_file models/sample_data_predictions.csv --generate_report
```

This will create various visualizations and an HTML report in the output directory.

## Recommended Workflow

For best results, follow this workflow:

1. **Prepare Your Data**: Ensure your conversations are formatted properly with both agent and caller utterances.

2. **Train Multiple Models**: Use the `--train_all` flag to train all six model architectures:
```bash
python main.py train --train_all --data_path your_data.csv --output_dir models/
```

3. **Evaluate Models**: Compare model performance to find the best one:
```bash
python main.py evaluate --evaluate_all --output_dir models/
```

4. **Make Predictions**: Use the best model for predictions:
```bash
python main.py predict --input_file new_conversations.csv --generate_report
```

5. **Review Reports**: Examine the generated HTML report to gain insights about complaints.

## Common Issues and Solutions

### Model Accuracy

If your model accuracy is low:
- Increase the training data size
- Increase the number of epochs
- Try a different model architecture
- Adjust the threshold (default is 0.5)

### Out of Memory Errors

If you encounter CUDA out of memory errors:
- Reduce batch size (`--batch_size 8`)
- Use a smaller model
- Process data in smaller chunks

### Slow Training

If training is slow:
- Use a GPU
- Reduce sequence length in config.py
- Use larger batch sizes if memory allows

## Next Steps

After getting familiar with the basic operation, consider:

1. **Fine-tuning hyperparameters**: Adjust learning rate, batch size, and model architecture.

2. **Creating a custom training script**: If you need more control over the training process.

3. **Deploying as a service**: The inference module can be integrated into a REST API.

4. **Extending visualizations**: Create custom visualizations for your specific needs. 