import tensorflow as tf
import numpy as np
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Load model
model = tf.keras.models.load_model("model/pneumonia_model.h5")

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return "PNEUMONIA"
    else:
        return "NORMAL"

# File picker (GUI)
Tk().withdraw()  # Hide main window
file_path = askopenfilename(title="Select X-ray Image")

if file_path:
    result = predict_image(file_path)
    print("\nSelected File:", file_path)
    print("Prediction:", result)
else:
    print("No file selected.")