import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import cv2

train_dir = r"C:\Users\Mohammed kaif M\OneDrive\Desktop\Brain tumor\datasets\training"
test_dir  = r"C:\Users\Mohammed kaif M\OneDrive\Desktop\Brain tumor\datasets\testing"

IMG_SIZE = 128
BATCH = 16

train_gen = ImageDataGenerator(rescale=1/255, rotation_range=20, zoom_range=0.2,width_shift_range=0.1, height_shift_range=0.1,horizontal_flip=True)

test_gen  = ImageDataGenerator(rescale=1/255)

train_data = train_gen.flow_from_directory(
    train_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH, class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    test_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH, class_mode='categorical',
    shuffle=False
)

#  model
base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base.trainable = False

x = Flatten()(base.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

# CALLBACKS
es = EarlyStopping(patience=3, restore_best_weights=True)
lr = ReduceLROnPlateau(factor=0.3, patience=2, min_lr=1e-6)

# TRAINING 
history = model.fit(
    train_data, validation_data=test_data,
    epochs=20, callbacks=[es, lr]
)

# ACCURACY GRAPH 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Training vs Validation Accuracy")
plt.show()

# CONFUSION MATRIX 
y_true = test_data.classes
y_pred = np.argmax(model.predict(test_data), axis=1)

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

model.save("Brain_tumor_final.h5")
print("Model saved successfully ✔")
