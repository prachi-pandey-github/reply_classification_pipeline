# Sentiment Analysis API

A Flask-based REST API that provides sentiment analysis using a fine-tuned DistilBERT model. The API classifies text as positive, negative, or neutral sentiment with confidence scores.

## Features

- **Single Text Prediction**: Analyze sentiment for individual text samples
- **Batch Prediction**: Process multiple texts in a single request
- **Pre-trained Model**: Uses a fine-tuned DistilBERT model for accurate sentiment classification
- **CORS Support**: Cross-origin requests enabled for web applications
- **Confidence Scores**: Returns prediction confidence along with sentiment labels

## Project Structure

```
models/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── test_api.py        # API testing script
├── README.md          # This file
└── results/           # Fine-tuned model directory
    ├── config.json
    ├── model.safetensors
    └── training_args.bin
```

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation & Setup

### 1. Clone or Download the Project

Navigate to your project directory:
```powershell
cd "C:\Users\prachi pandey\Desktop\models"
```

### 2. Create Virtual Environment (Recommended)

```powershell
python -m venv .venv
```

### 3. Activate Virtual Environment

```powershell
.venv\Scripts\Activate.ps1
```

If you encounter execution policy issues, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

## Running the API

### Start the Server

```powershell
python app.py
```

The API will start running on `http://127.0.0.1:5000` by default.

You should see output similar to:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

## API Endpoints

### 1. Single Text Prediction

**Endpoint**: `POST /predict`

**Request Body**:
```json
{
    "text": "Your text to analyze"
}
```

**Response**:
```json
{
    "label": "positive|negative|neutral",
    "confidence": 0.95
}
```

**Example using curl**:
```powershell
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{\"text\": \"I love this product!\"}'
```

### 2. Batch Text Prediction

**Endpoint**: `POST /batch_predict`

**Request Body**:
```json
{
    "texts": ["First text to analyze", "Second text to analyze"]
}
```

**Response**:
```json
[
    {
        "text": "First text to analyze",
        "label": "neutral",
        "confidence": 0.87
    },
    {
        "text": "Second text to analyze", 
        "label": "positive",
        "confidence": 0.92
    }
]
```

## Testing the API

### Using the Test Script

Run the included test script:
```powershell
python test_api.py
```

### Using Python Requests

```python
import requests

# Single prediction
response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={"text": "I feel great today!"}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://127.0.0.1:5000/batch_predict",
    json={"texts": ["I love this!", "This is terrible."]}
)
print(response.json())
```

### Using Postman or Similar Tools

1. Set method to `POST`
2. Set URL to `http://127.0.0.1:5000/predict` or `http://127.0.0.1:5000/batch_predict`
3. Set Content-Type header to `application/json`
4. Add JSON body as shown in the examples above

## Model Information

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Task**: Sentiment Classification
- **Labels**: 
  - `positive`: Positive sentiment
  - `negative`: Negative sentiment  
  - `neutral`: Neutral sentiment
- **Model Location**: `./results/` directory

## Troubleshooting

### Common Issues

1. **Port already in use**: If port 5000 is occupied, modify `app.py` to use a different port:
   ```python
   app.run(debug=True, port=5001)
   ```

2. **Module not found**: Ensure virtual environment is activated and dependencies are installed:
   ```powershell
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **Model loading errors**: Verify that the `results/` directory contains the trained model files.

4. **CORS issues**: The API includes CORS support. If you still encounter issues, check that `flask-cors` is installed.

### Development Mode

The API runs in debug mode by default, which provides:
- Automatic reloading when code changes
- Detailed error messages
- Interactive debugger

For production deployment, disable debug mode:
```python
app.run(debug=False)
```

## Dependencies

- **Flask**: Web framework for the API
- **transformers**: Hugging Face library for the pre-trained model
- **torch**: PyTorch for model inference
- **flask-cors**: Cross-Origin Resource Sharing support

## License

This project is for educational/research purposes. Please ensure compliance with model licensing terms when using in production.
