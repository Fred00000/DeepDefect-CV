import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "defect_model.h5"   # model path
IMG_PATH = "../data/val/crazing/crazing_241.jpg"
IMG_SIZE = (224, 224)

PATCH_SIZE = 32
STRIDE = 16

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]

# -----------------------------
# Load model
# -----------------------------
model = load_model(MODEL_PATH)

# -----------------------------
# Load & preprocess image (CORRECT)
# -----------------------------
img_pil = image.load_img(IMG_PATH, target_size=IMG_SIZE)
img_np = image.img_to_array(img_pil)

# MobileNetV2 preprocessing
img_pre = preprocess_input(img_np.copy())
input_img = np.expand_dims(img_pre, axis=0)

# -----------------------------
# Base prediction
# -----------------------------
preds = model.predict(input_img, verbose=0)
class_idx = int(np.argmax(preds[0]))
base_score = float(preds[0][class_idx])

print(f"Predicted class: {CLASS_NAMES[class_idx]}")
print(f"Confidence: {base_score:.4f}")

# -----------------------------
# Occlusion Sensitivity
# -----------------------------
heatmap = np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.float32)

for y in range(0, IMG_SIZE[0] - PATCH_SIZE, STRIDE):
    for x in range(0, IMG_SIZE[1] - PATCH_SIZE, STRIDE):
        occluded = img_pre.copy()

        # neutral value for MobileNetV2 space
        occluded[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :] = 0.0

        occluded_input = np.expand_dims(occluded, axis=0)
        occluded_pred = model.predict(occluded_input, verbose=0)
        occluded_score = occluded_pred[0][class_idx]

        drop = base_score - occluded_score
        heatmap[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += drop

# -----------------------------
# Normalize heatmap safely
# -----------------------------
heatmap = np.maximum(heatmap, 0)
heatmap /= (np.max(heatmap) + 1e-8)

# -----------------------------
# Overlay heatmap
# -----------------------------
heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
heatmap_uint8 = np.uint8(255 * heatmap_resized)
heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

original_img = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
overlay = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

# -----------------------------
# Display result
# -----------------------------
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title(f"Occlusion Sensitivity: {CLASS_NAMES[class_idx]}")
plt.axis("off")
plt.show()
