# ------------------------------------------------------------
# Brain Tumor Classification with EfficientNetB0 (TensorFlow)
# ------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt

print("TensorFlow Version:", tf.__version__)

# GPU Configuration
gpus = tf.config.list_physical_devices("GPU")
print("GPU Available:", gpus)

if gpus:
    try:
        # Set GPU memory growth to avoid OOM errors
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"⚠️ {e}")
else:
    print("⚠️ No GPU detected. Model will train on CPU.")
    print("💡 To fix: Install CUDA and cuDNN, then reinstall TensorFlow with: pip install tensorflow[and-cuda]")

# ------------------------------------------------------------
# 1. DATASET PATHS
# ------------------------------------------------------------
train_dir = r"C:\Users\Mohammed kaif M\OneDrive\Desktop\Brain tumor\datasets\training"
test_dir  = r"C:\Users\Mohammed kaif M\OneDrive\Desktop\Brain tumor\datasets\testing"

print("Train Dir:", train_dir)
print("Test Dir:", test_dir)

# ------------------------------------------------------------
# 2. LOAD DATASETS
# ------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)

num_classes = len(train_ds.class_names)
print("Detected Classes:", train_ds.class_names)

# Performance boost
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

# ------------------------------------------------------------
# 3. DATA AUGMENTATION
# ------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# ------------------------------------------------------------
# 4. EFFICIENTNETB0 TRANSFER LEARNING
# ------------------------------------------------------------
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze for stability

# ------------------------------------------------------------
# 5. BUILD THE MODEL
# ------------------------------------------------------------
inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)  # EfficientNet likes higher dropout

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------------------------------------------------
# 6. TRAIN MODEL
# ------------------------------------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
)   

# ------------------------------------------------------------
# 7. SAVE TRAINED MODEL
# ------------------------------------------------------------
save_path = r"C:\Users\Mohammed kaif M\OneDrive\Desktop\BrainTumorProject\brain_tumor_efficientb0.h5"
model.save(save_path)
print("Model saved at:", save_path)

# ------------------------------------------------------------
# 8. TRAINING CURVES
# ------------------------------------------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")
plt.legend(["Train", "Validation"])

plt.subplot(1,2,2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.legend(["Train", "Validation"])

plt.show()
