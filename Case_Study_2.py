import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import warnings

# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress Python warnings (optional, e.g., for Keras deprecations)
warnings.filterwarnings('ignore')

# Load CSV with image paths and labels
try:
    data = pd.read_csv("csv_files\\waste_data.csv")
    print(f"Loaded {len(data)} samples from waste_data.csv")
except FileNotFoundError:
    print("Error: 'csv_files\\waste_data.csv' not found. Please check the file path.")
    exit(1)

# Load images and labels
def load_images(df, target_size=(64, 64)):
    images = []
    labels = []
    for _, row in df.iterrows():
        try:
            img = load_img(row['image_path'], target_size=target_size)
            img_array = img_to_array(img) / 255.0  # Normalize
            images.append(img_array)
            labels.append(row['label'])
        except FileNotFoundError:
            print(f"Warning: Image not found at {row['image_path']}. Skipping.")
    if not images:
        raise ValueError("No valid images loaded. Check your dataset.")
    return np.array(images), tf.keras.utils.to_categorical(labels, num_classes=4)

# Load data
try:
    X, y = load_images(data)
    print(f"Successfully loaded {len(X)} images.")
except ValueError as e:
    print(e)
    exit(1)

# Dynamic split based on dataset size
if len(X) < 10:
    print("Warning: Dataset is very small (<10 samples). Results may be unreliable.")
train_size = int(0.8 * len(X))  # 80% for training
if train_size == 0 or len(X) - train_size == 0:
    print("Error: Dataset too small to split. Need at least 2 samples.")
    exit(1)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Build CNN model with explicit Input layer
model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),  # Explicit input layer to avoid warning
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')  # 4 classes
])

# Compile and train with dynamic batch size
batch_size = min(32, len(X_train))  # Use smaller batch size if dataset is small
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=batch_size, validation_data=(X_test, y_test))

# Predict on new image with error handling
new_image_path = 'sample_image.jpg'  # Update to a valid path
try:
    new_image = load_img(new_image_path, target_size=(64, 64))
    new_image_array = img_to_array(new_image) / 255.0
    prediction = model.predict(np.expand_dims(new_image_array, axis=0))
    print(f"Predicted class: {np.argmax(prediction)} (0=plastic, 1=glass, 2=metal, 3=paper)")
except FileNotFoundError:
    print(f"Error: Prediction image not found at {new_image_path}")