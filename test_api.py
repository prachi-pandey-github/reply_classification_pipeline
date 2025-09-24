import requests

# Single prediction
res = requests.post(
    "http://127.0.0.1:5000/predict",
    json={"text": "I feel great today!"}
)
print(res.json())

# Batch prediction
res = requests.post(
    "http://127.0.0.1:5000/batch_predict",
    json={"texts": ["I feel great today!", "This is bad."]}
)
print(res.json())
