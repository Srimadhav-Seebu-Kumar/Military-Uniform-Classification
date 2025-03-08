import cv2
import numpy as np
import os
from dataset_preprocessing import load_images, CATEGORIES

# Initialize HOG + SVM detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_humans_hog(image):
    """Detect humans using HOG+SVM."""
    image_resized = cv2.resize(image, (640, 480))
    boxes, _ = hog.detectMultiScale(image_resized, winStride=(4,4), padding=(8,8), scale=1.05)
    return image_resized, boxes

def extract_humans(image, boxes, padding=20):
    """Extract detected human figures with extra padding around the bounding box."""
    cropped_images = []
    h, w, _ = image.shape  # Get image dimensions
    
    for (x, y, bw, bh) in boxes:
        # Expand bounding box by padding (but keep it inside image boundaries)
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + bw + padding, w)
        y2 = min(y + bh + padding, h)
        
        # Crop and resize
        cropped = image[y1:y2, x1:x2]
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            cropped_images.append(cv2.resize(cropped, (224, 224)))  # Resize for CNN
    
    return cropped_images

if __name__ == "__main__":
    data, labels = load_images()
    
    extracted_path = "extracted_humans"
    os.makedirs(extracted_path, exist_ok=True)
    
    for category in CATEGORIES:
        os.makedirs(os.path.join(extracted_path, category), exist_ok=True)

    for i, img in enumerate(data):
        detected_img, boxes = detect_humans_hog(img)
        cropped_humans = extract_humans(img, boxes, padding=50)  # Increase padding

        if cropped_humans:
            for j, human in enumerate(cropped_humans):
                cv2.imwrite(os.path.join(extracted_path, labels[i], f"{i}_{j}.jpg"), human)

    print("Human detection and extraction complete.")
