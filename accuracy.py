from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your trained model
model = load_model("model_best_final_1.h5")

# Test data directory
test_dir = r"C:\Users\Mohammed kaif M\OneDrive\Desktop\Brain tumor\datasets\training"

# ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predictions
pred = model.predict(test_generator)
y_pred = np.argmax(pred, axis=1)

# True labels
y_true = test_generator.classes

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

# ================================
# 📊 1 — Metric Bar Chart
# ================================
metrics = [accuracy, precision, recall, f1]
names = ["Accuracy", "Precision", "Recall", "F1 Score"]

plt.figure(figsize=(7, 5))
plt.bar(names, metrics)
plt.title("Evaluation Metrics")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.grid(axis='y')
plt.show()

# ================================
# 📊 2 — Confusion Matrix Heatmap
# ================================
cm = confusion_matrix(y_true, y_pred)
class_labels = list(test_generator.class_indices.keys())

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
