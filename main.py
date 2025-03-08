import os
import sys
import cv2
import uuid  # To generate unique filenames
import requests
from src.predict import predict_uniform

# Ensure Python can find 'src/' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Hugging Face model URL
MODEL_URL = "https://huggingface.co/Idlebeing/Military-Uniform-Model/resolve/main/uniform_classifier.h5"
MODEL_PATH = "models/uniform_classifier.h5"

# Function to download model if not found
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Hugging Face...")
        os.makedirs("models", exist_ok=True)  # Ensure directory exists
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete!")

# Download the model if necessary
download_model()

# Define image path (Change this to your test image)
test_image = "images (12).jpg"

# Ensure the file exists
if not os.path.exists(test_image):
    print(f"Error: The file {test_image} does not exist.")
    sys.exit(1)

# Run prediction
predicted_class = predict_uniform(test_image)
print(f"Predicted Category: {predicted_class}")
