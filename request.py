import requests

# Single prediction
response = requests.post("http://localhost:5000/predict", 
                        json={"text": "I love this service!"})
print(response.json())

# Batch prediction
texts = ["Excellent!", "Not good", "Neutral response"]
response = requests.post("http://localhost:5000/batch_predict", 
                        json={"texts": texts})
print(response.json())