import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

model = tf.keras.models.load_model("models/uniform_classifier.h5")

CATEGORIES = ["Airforce", "Army", "Navy"]
label_encoder = LabelEncoder()
label_encoder.fit(CATEGORIES)

def predict_uniform(image_path):
    """Predicts the military uniform category and confidence score."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)  

    prediction = model.predict(image)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100 
    class_name = label_encoder.inverse_transform([class_idx])[0]

    return class_name, confidence


