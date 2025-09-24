from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask_cors import CORS
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)

MODEL_PATH = "results"
BASE_MODEL = "distilbert-base-uncased"  # Original model for tokenizer


LABEL_MAPPING = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral", 
    "LABEL_2": "positive",
}

# Load tokenizer from base model and fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        confidence, pred_class = torch.max(probabilities, dim=-1)
        
        # Get label name from model config and map to sentiment
        raw_label = model.config.id2label[pred_class.item()]
        label = LABEL_MAPPING.get(raw_label, raw_label)

    return jsonify({
        "label": label,
        "confidence": float(confidence.item())
    })

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    data = request.get_json()
    texts = data.get("texts", [])

    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        confidences, pred_classes = torch.max(probabilities, dim=-1)
        
        # Convert to sentiment labels
        raw_labels = [model.config.id2label[pred.item()] for pred in pred_classes]
        labels = [LABEL_MAPPING.get(raw_label, raw_label) for raw_label in raw_labels]

    return jsonify([
        {"text": t, "label": label, "confidence": float(conf.item())}
        for t, label, conf in zip(texts, labels, confidences)
    ])

if __name__ == "__main__":
    app.run(debug=True)


