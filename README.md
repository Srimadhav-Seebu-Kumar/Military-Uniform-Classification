# Military Uniform Classification  

This project classifies military uniforms into different categories using a trained deep learning model. It utilizes computer vision techniques to analyze images and predict the uniform type (Airforce, Army, or Navy).  

## Features  
- Deep learning-based uniform classification  
- Pretrained model hosted on Hugging Face (automatically downloads if missing)  
- Simple API for prediction  
- Supports custom image input  

---

## Project Structure  
```
Military-Uniform-Classification/
│── dataset/                   # Raw dataset images  
│   ├── Airforce/  
│   ├── Army/  
│   ├── Navy/  
│── extracted_humans/           # Processed dataset with extracted humans  
│   ├── Airforce/  
│   ├── Army/  
│   ├── Navy/  
│── models/                     # Model storage (Downloaded automatically)  
│── src/                        # Source code  
│   ├── dataset_preprocessing.py  
│   ├── human_detection.py  
│   ├── predict.py               # Model inference script  
│   ├── train_classifier.py       # Model training script  
│── main.py                     # Script to test the model  
│── requirements.txt             # Python dependencies  
│── README.md                    # Project documentation  
```

## Installation  

### Step 1: Clone the Repository  
```bash
git clone https://github.com/Srimadhav-Seebu-Kumar/Military-Uniform-Classification.git
cd Military-Uniform-Classification
```

### Step 2: Set Up a Virtual Environment (Optional but Recommended)  
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### Step 3: Install Dependencies  
```bash
pip install -r requirements.txt
```

---

## Model Download  
The model is not stored in this repository due to GitHub's file size limits.  
It will automatically download from Hugging Face when `main.py` is executed.  

Manual Download (if needed):  
[Download Model from Hugging Face](https://huggingface.co/Idlebeing/Military-Uniform-Model/resolve/main/uniform_classifier.h5)  

If downloading manually, place the model in the `models/` directory:  
```
models/uniform_classifier.h5
```

---

## Usage  

### Running the Model on an Image  
To classify a military uniform, place an image in the project folder and run:  
```bash
python main.py
```

By default, the script uses `images (12).jpg`.  
To test a different image, modify `test_image` in `main.py`:

```python
test_image = "your_image.jpg"
```

### Example Output  
```
Downloading model from Hugging Face...
Download complete!
Predicted Category: Army
```

---

## Training the Model  
To train the classifier on your dataset:  
```bash
python src/train_classifier.py
```

Ensure the dataset is structured as:  
```
dataset/
├── Airforce/
├── Army/
├── Navy/
```

---

## Troubleshooting  

- **ModuleNotFoundError:** Install dependencies using:  
  ```bash
  pip install -r requirements.txt
  ```
- **Model not downloading automatically?** Manually download it from Hugging Face.  
- **Incorrect predictions?** Ensure the image clearly displays the uniform without obstructions.  

---

## License  
This project is open-source and available under the MIT License.  

---

## Author  
Developed by Srimadhav Seebu Kumar.  
Contributions are welcome through pull requests.  

