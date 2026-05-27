import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ---------------------------------
# CONFIGURATION
# ---------------------------------
MODEL_PATH = "model_best_final_1.h5"

TEST_DIR = r"C:\Users\Mohammed kaif M\OneDrive\Desktop\Brain tumor\datasets\training"

IMAGE_SIZE = (224, 224)

class_map = {
    0: 'glioma_tumor',
    1: 'meningioma_tumor',
    2: 'no_tumor',
    3: 'pituitary_tumor'
}

# ---------------------------------
# LOAD MODEL
# ---------------------------------
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!\n")

# ---------------------------------
# IMAGE PREPROCESSING
# ---------------------------------
def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# ---------------------------------
# ACCURACY TESTING
# ---------------------------------
correct = 0
total = 0

print("Testing accuracy...\n")

for label_index, label_name in class_map.items():
    folder = os.path.join(TEST_DIR, label_name)

    if not os.path.exists(folder):
        print(f"❌ Missing folder: {folder}")
        continue

    for image_name in os.listdir(folder):
        img_path = os.path.join(folder, image_name)
        img_input = preprocess_image(img_path)

        if img_input is None:
            continue

        preds = model.predict(img_input, verbose=0)
        predicted = int(np.argmax(preds))

        total += 1
        if predicted == label_index:
            correct += 1

# ---------------------------------
# RESULT OUTPUT
# ---------------------------------
if total == 0:
    print("No test images found. Check folder path!")
else:
    accuracy = correct / total * 100
    print("==============================")
    print(f"Total Images          : {total}")
    print(f"Correct Predictions   : {correct}")
    print(f"Accuracy              : {accuracy:.2f}%")
    print("==============================")
