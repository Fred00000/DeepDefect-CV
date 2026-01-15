import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]

HIGH_CONF = 0.70
LOW_CONF  = 0.50

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_trained_model():
    return load_model("defect_model.h5")

model = load_trained_model()

# -----------------------------
# Occlusion Sensitivity
# -----------------------------
def occlusion_sensitivity(img_array, class_idx, patch=32, stride=16):
    heatmap = np.zeros((224, 224), dtype=np.float32)

    base_score = model.predict(img_array, verbose=0)[0][class_idx]
    img = img_array[0].copy()

    for y in range(0, 224 - patch, stride):
        for x in range(0, 224 - patch, stride):
            occluded = img.copy()
            occluded[y:y+patch, x:x+patch, :] = 0.0  # neutral value for MobileNetV2

            occluded_input = np.expand_dims(occluded, axis=0)
            score = model.predict(occluded_input, verbose=0)[0][class_idx]

            heatmap[y:y+patch, x:x+patch] += (base_score - score)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    return heatmap

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Industrial Defect Detection", layout="centered")

st.title("Industrial Surface Defect Detection")
st.write(
    "Upload a steel surface image to classify defects.\n\n"
    "The system uses confidence-based decision logic to avoid unreliable predictions."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded Image", width=500)

    # -----------------------------
    # Preprocessing (CORRECT)
    # -----------------------------
    img = image_pil.resize(IMG_SIZE)
    img_np = np.array(img)
    img_pre = preprocess_input(img_np)
    img_array = np.expand_dims(img_pre, axis=0)

    # -----------------------------
    # Prediction
    # -----------------------------
    preds = model.predict(img_array)
    class_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][class_idx])

    st.subheader("Prediction Result")

    # -----------------------------
    # Decision Logic
    # -----------------------------
    if confidence >= HIGH_CONF:
        st.success("High confidence prediction")
        st.write(f"**Defect Type:** {CLASS_NAMES[class_idx]}")
        st.write(f"**Confidence:** {confidence:.3f}")

        # Explainability
        st.subheader("Occlusion Sensitivity Explanation")
        heatmap = occlusion_sensitivity(img_array, class_idx)

        heatmap = cv2.resize(heatmap, IMG_SIZE)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        st.image(overlay, caption="Model Attention (Occlusion Sensitivity)", width=500)

    elif LOW_CONF <= confidence < HIGH_CONF:
        st.warning("Medium confidence prediction")
        st.write(
            "The image partially matches known defect patterns.\n\n"
            "Prediction should be verified by a human inspector."
        )
        st.write(f"**Proposed Defect:** {CLASS_NAMES[class_idx]}")
        st.write(f"**Confidence:** {confidence:.3f}")

    else:
        st.error("Prediction rejected")
        st.write(
            "The uploaded image does not confidently match any known defect class.\n\n"
            "**Possible reasons:**\n"
            "- Image is not a steel surface\n"
            "- Defect type not present in training data\n"
            "- Poor lighting or image quality\n\n"
            "**Manual inspection recommended.**"
        )
