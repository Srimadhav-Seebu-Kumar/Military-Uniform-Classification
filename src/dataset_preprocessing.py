import os
import cv2
import numpy as np

# Define dataset path
DATASET_PATH = "dataset"
CATEGORIES = ["Airforce", "Army", "Navy"]

def load_images():
    """Load images from dataset, resize to 640x480."""
    data, labels = [], []
    for category in CATEGORIES:
        folder_path = os.path.join(DATASET_PATH, category)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (640, 480))
                data.append(img)
                labels.append(category)
    return np.array(data), np.array(labels)

if __name__ == "__main__":
    data, labels = load_images()
    print(f"Loaded {len(data)} images.")
