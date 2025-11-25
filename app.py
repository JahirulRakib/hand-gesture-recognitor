import streamlit as st
import numpy as np
import cv2
import joblib
from pathlib import Path
from skimage import color, transform  # for same preprocessing as training

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="Hand Gesture Recognition (SVM)",
    page_icon="✋",
    layout="centered"
)

st.title("✋ Hand Gesture Recognition (SVM)")
st.write("Model: **svm_final.pkl** · Classes: A–Z, SPACE, DELETE, NOTHING")

# -------------------------
# CLASS LABELS (for info only)
# -------------------------
CLASS_LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "SPACE", "DELETE", "NOTHING"
]

# -------------------------
# LOAD MODEL WITH JOBLIB
# -------------------------
@st.cache_resource
def load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        st.error(f"Model file not found: {path.resolve()}")
        st.stop()
    try:
        model = joblib.load(path)
    except Exception as e:
        st.error(f"❌ Failed to load model with joblib: {e}")
        st.stop()
    return model

model = load_model("svm_final.pkl")

# -------------------------
# IMAGE PREPROCESSING (MATCH TRAINING)
# -------------------------
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Matches your training code:

        gray = color.rgb2gray(img)
        small = transform.resize(gray, (30,30))
        X_small.append(small.flatten())

    Streamlit gives `image` as RGB uint8 (0–255).
    We convert to float [0,1], then apply same steps.
    """
    # image: RGB, uint8 [0..255]
    img_float = image.astype("float32") / 255.0

    # rgb2gray expects float image
    gray = color.rgb2gray(img_float)

    # resize to 30x30
    small = transform.resize(gray, (30, 30), anti_aliasing=True)

    # flatten to 900-dim vector
    features = small.flatten().astype("float32")

    # shape (1, n_features)
    return features.reshape(1, -1)

def predict_gesture(image: np.ndarray):
    features = preprocess_image(image)
    # pipeline: PCA -> SVC, so just predict
    label = model.predict(features)[0]
    # we don't have predict_proba (SVC(probability=False) by default)
    return label, None

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("How to use")
st.sidebar.markdown(
"""
1. Choose **Upload Image** or **Use Camera**  
2. Show a clear hand gesture (A–Z, SPACE, DELETE, NOTHING)  
3. Press **Predict Gesture**

This app uses an SVM (with PCA) trained on 30×30 grayscale images.
"""
)

# -------------------------
# MAIN UI
# -------------------------
mode = st.radio("Input mode", ["Upload Image", "Use Camera"])

uploaded_image = None

if mode == "Upload Image":
    file = st.file_uploader("Upload a hand gesture image", type=["jpg", "jpeg", "png"])
    if file is not None:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        uploaded_image = img_rgb
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

else:  # Use Camera
    camera_img = st.camera_input("Capture a hand gesture")
    if camera_img is not None:
        file_bytes = np.asarray(bytearray(camera_img.getvalue()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        uploaded_image = img_rgb
        st.image(uploaded_image, caption="Captured Image", use_container_width=True)

# -------------------------
# PREDICTION
# -------------------------
if uploaded_image is not None:
    if st.button("Predict Gesture"):
        with st.spinner("Predicting..."):
            label, _ = predict_gesture(uploaded_image)

        st.subheader("Prediction")
        st.markdown(f"### 👉 {label}")

        if label in ["SPACE", "DELETE", "NOTHING"]:
            st.info(f"Special command detected: **{label}**")
else:
    st.info("Please upload or capture an image to start.")
