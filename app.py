from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None
# Standard binary sentiment classification (distilbert-base-uncased has 2 classes)
label_map = {0: 'negative', 1: 'positive'}

def clean_text(text):
    """Clean text by removing non-alphabetic characters and standardizing whitespace"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_model():
    """Load the fine-tuned model and tokenizer"""
    global model, tokenizer, device
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Load the pre-trained DistilBERT model for sequence classification
        # Using a model specifically trained for sentiment analysis
        model_path = "distilbert-base-uncased-finetuned-sst-2-english"  # SST-2 dataset (binary sentiment)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def predict_sentiment(text):
    """Predict sentiment for a single text"""
    if model is None or tokenizer is None:
        return {"error": "Model not loaded"}
    
    try:
        # Clean and tokenize text
        cleaned_text = clean_text(text)
        inputs = tokenizer(
            cleaned_text, 
            truncation=True, 
            padding='max_length', 
            max_length=512, 
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process results
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][predicted_label].item()
        
        return {
            "text": text,
            "sentiment": label_map[predicted_label],
            "confidence": round(confidence, 4),
            "all_probabilities": {
                label_map[i]: round(probabilities[0][i].item(), 4) 
                for i in range(len(label_map))
            }
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.route('/')
def home():
    return jsonify({
        "message": "Sentiment Analysis API", 
        "status": "active",
        "endpoints": {
            "/health": "Check API health",
            "/predict": "Predict sentiment (POST)",
            "/batch_predict": "Batch prediction (POST)"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        "status": "healthy",
        "model": model_status,
        "device": str(device)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment for a single text"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text']
        
        if not text or not text.strip():
            return jsonify({"error": "Text cannot be empty"}), 400
        
        result = predict_sentiment(text)
        
        if 'error' in result:
            return jsonify(result), 500
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict sentiment for multiple texts"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({"error": "No texts provided"}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({"error": "Texts must be a list"}), 400
        
        if len(texts) > 100:  # Limit batch size
            return jsonify({"error": "Batch size too large (max 100)"}), 400
        
        results = []
        for text in texts:
            if text and text.strip():
                result = predict_sentiment(text)
                results.append(result)
            else:
                results.append({"text": text, "error": "Empty text"})
        
        return jsonify({
            "count": len(results),
            "results": results
        })
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/examples', methods=['GET'])
def examples():
    """Provide example predictions"""
    example_texts = [
        "This is a great product!",
        "I'm not interested.",
        "Tell me more about the features.",
        "Can we schedule a demo?",
        "This is terrible."
    ]
    
    results = []
    for text in example_texts:
        result = predict_sentiment(text)
        results.append(result)
    
    return jsonify({
        "examples": results
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("Loading model...")
    if load_model():
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to load model. Exiting.")