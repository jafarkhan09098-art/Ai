
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image

print("Downloading Cat-Dog Dataset...")

# Load dataset
raw_data = tfds.load(
    "cats_vs_dogs",
    split="train[:2500]",
    as_supervised=True
)

# Prepare training and validation data
train_raw = raw_data.take(2000)
val_raw = raw_data.skip(2000).take(500)

def prepare_image(image, label):
    image = tf.image.resize(image, (180, 180))
    image = image / 255.0
    return image, label

train_data = train_raw.map(prepare_image).batch(32)
val_data = val_raw.map(prepare_image).batch(32)

# Create CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(180, 180, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
print("Training Started...")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=3
)

# Save model
model.save("model.keras")

# Create test.jpg from a validation image
for image, label in val_raw.take(1):
    Image.fromarray(image.numpy()).save("test.jpg")

print("\nModel Saved Successfully!")
print("Test Image Created Successfully!")
