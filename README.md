# Brain Tumor Classification System

## Project Overview
The **Brain Tumor Classification System** is an AI-powered application designed to automatically detect and classify brain tumors from MRI scans. Early detection and accurate classification of brain tumors can significantly improve patient treatment planning and outcomes. This system classifies MRI images into the following categories:

- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor (Healthy)**

The project uses **Convolutional Neural Networks (CNNs)** and **transfer learning** techniques to achieve high accuracy in tumor detection.

---

## Features
- Upload MRI scans and classify them in real-time.
- Multi-class classification of brain tumors with **~90% accuracy**.
- Easy-to-use web interface (Flask-based).
- Optional visual feedback using heatmaps (Grad-CAM) to show the tumor location.
- Supports MRI images of varying sizes through preprocessing and normalization.

---

## Dataset
- Publicly available brain MRI datasets (e.g., [Kaggle Brain Tumor Dataset](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection))
- Dataset contains labeled images for:
  - Glioma
  - Meningioma
  - Pituitary Tumor
  - No Tumor
- Dataset is split into:
  - Training: 70%
  - Validation: 15%
  - Testing: 15%

---

## Technology Stack
- **Programming Language:** Python 3.x
- **Deep Learning:** TensorFlow, Keras
- **Image Processing:** OpenCV, Pillow (PIL)
- **Web Framework:** Flask
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Flask + HTML/CSS/JS

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Brain-Tumor-Classification.git
cd Brain-Tumor-Classification
pip install -r requirements.txt

If the model file was downloaded from GitHub as a small pointer file, install Git LFS and run:

```bash
git lfs pull
```

TensorFlow requires a supported Python version. Use Python 3.10-3.12 for this project, not Python 3.14.


Run the Flask application

python app.py


Open your browser

http://127.0.0.1:5000/


Upload MRI images to classify brain tumors.

View results including predicted tumor type and probability.

Model Training

If you want to train your own model:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


Training: Use model.fit() with preprocessed dataset.

Evaluation: Use model.evaluate() on test data.

Expected Accuracy: ~90% on test data.

Results

Accuracy with CNN model: ~90%

Confusion matrix and classification report included for detailed analysis.

Project Structure
Brain-Tumor-Classification/
│
├─ backend/
│  ├─ app.py              # Flask backend
│  ├─ model_best_final_1.h5  # Trained detection model
│  └─ utils.py            # Preprocessing and helper functions
│
├─ frontend/
│  ├─ templates/
│  │  └─ index.html       # Web interface
│  └─ static/
│     ├─ css/
│     └─ js/
│
├─ dataset/               # MRI images
│  ├─ Train/
│  ├─ Validation/
│  └─ Test/
│
├─ requirements.txt
└─ README.md

Future Enhancements

Tumor segmentation to highlight affected regions.

Integration with hospital PACS systems for automated analysis.

Multi-modal analysis combining MRI with clinical data.

Deploy on cloud platforms for real-time remote diagnosis.

References

Kaggle Brain MRI Dataset: https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection

TensorFlow Documentation: https://www.tensorflow.org/

Keras Documentation: https://keras.io/

License

This project is licensed under the MIT License.

Author: Mohammed Kaif M
Department: AIML, Sri Krishna Institute of Technology, Bengaluru, India

<img width="640" height="480" alt="Figure_2" src="https://github.com/user-attachments/assets/13169343-14be-4879-b30d-e60e88977b57" />
<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/424eeaff-e788-4f77-95a8-1334cb42fa7c" />
