
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("model.keras")

# Class names
classes = ["cats", "dogs"]

# Load test image
img = tf.keras.utils.load_img(
    "test.jpg",
    target_size=(180, 180)
)

img = tf.keras.utils.img_to_array(img)
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img, verbose=0)

result = classes[np.argmax(prediction)]

print("Predicted Class:", result)
print("Confidence:", round(np.max(prediction) * 100, 2), "%")
