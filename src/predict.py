import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = tf.keras.models.load_model("models/uniform_classifier.h5")

# Manually define label encoder instead of importing from train_classifier.py
CATEGORIES = ["Airforce", "Army", "Navy"]
label_encoder = LabelEncoder()
label_encoder.fit(CATEGORIES)

def predict_uniform(image_path):
    """Predicts the military uniform category and confidence score."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Reshape for model input

    # Run prediction
    prediction = model.predict(image)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100  # Convert confidence to percentage
    class_name = label_encoder.inverse_transform([class_idx])[0]

    return class_name, confidence


