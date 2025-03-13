import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import cv2

EXTRACTED_PATH = "extracted_humans"
CATEGORIES = ["Airforce", "Army", "Navy"]

def load_extracted_data():
    """Loads extracted human images for training."""
    data, labels = [], []
    for category in CATEGORIES:
        folder_path = os.path.join(EXTRACTED_PATH, category)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                data.append(img)
                labels.append(category)
    return np.array(data), np.array(labels)

if __name__ == "__main__":  
    
    data, labels = load_extracted_data()
    data = data / 255.0  

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_encoded = tf.keras.utils.to_categorical(labels_encoded)

    X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

    
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

    model.save("models/uniform_classifier.h5")
    print("Model training complete and saved.")
