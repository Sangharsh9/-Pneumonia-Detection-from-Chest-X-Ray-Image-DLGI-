import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load model
model = tf.keras.models.load_model("model/pneumonia_model.h5")

st.title("🩺 Pneumonia Detection System (Advanced)")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg","png","jpeg"])


# -----------------------------
# ✅ FIXED Grad-CAM (for MobileNet)
# -----------------------------
def get_gradcam_heatmap(img_array, model):
    try:
        base_model = model.layers[0]

        # find last conv layer inside base model
        last_conv_layer = None
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[last_conv_layer.output, model.layers[-1].output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

        return heatmap.numpy()

    except:
        return None


# -----------------------------
# MAIN LOGIC
# -----------------------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ✅ Better preprocessing
    img_resized = img.resize((224,224))
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    # ✅ Improved threshold
    threshold = 0.4

    if prediction > threshold:
        confidence = prediction * 100
        st.error(f"❌ Pneumonia Detected ({confidence:.2f}%)")
    else:
        confidence = (1 - prediction) * 100
        st.success(f"✅ Normal ({confidence:.2f}%)")

    st.progress(int(confidence))

    # -----------------------------
    # Grad-CAM
    # -----------------------------
    st.subheader("🔥 Heatmap (Infected Area)")

    heatmap = get_gradcam_heatmap(img_array, model)

    if heatmap is None or heatmap.size == 0:
        st.warning("⚠️ Heatmap not available")
    else:
        heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = cv2.addWeighted(
            np.array(img), 0.6, heatmap, 0.4, 0
        )

        st.image(superimposed_img, caption="Grad-CAM Result", use_column_width=True)