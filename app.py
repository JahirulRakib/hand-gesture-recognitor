import streamlit as st
import numpy as np
import cv2
import joblib
from pathlib import Path
from skimage import color, transform
import zipfile

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Hand Gesture Recognition (SVM)",
    page_icon="✋",
    layout="centered"
)

st.title("✋ Hand Gesture Recognition (SVM)")
st.write("Model: **svm_final.pkl** · Classes: A–Z, SPACE, DELETE, NOTHING")

# =========================
# FILE PATHS
# =========================
MODEL_PKL = Path("svm_final.pkl")
MODEL_ZIP = Path("svm_final.zip")

# =========================
# CLASS LABELS (info only)
# =========================
CLASS_LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "SPACE", "DELETE", "NOTHING"
]

# =========================
# MODEL LOADING
# =========================
def ensure_model_unzipped():
    """
    If svm_final.pkl doesn't exist but svm_final.zip does,
    extract svm_final.pkl from the zip.

    Make sure your zip contains a file literally named 'svm_final.pkl'
    at the top level.
    """
    if not MODEL_PKL.exists():
        if not MODEL_ZIP.exists():
            raise FileNotFoundError(
                "Neither 'svm_final.pkl' nor 'svm_final.zip' found in the app directory."
            )
        # Extract only svm_final.pkl from the zip
        with zipfile.ZipFile(MODEL_ZIP, "r") as zf:
            zf.extract("svm_final.pkl")  # name inside the zip must match exactly

@st.cache_resource
def load_model(model_path: str):
    # If pkl not present but zip is, unzip first
    ensure_model_unzipped()

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

# =========================
# PREPROCESSING
# =========================
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Match your training code:

        gray = color.rgb2gray(img)
        small = transform.resize(gray, (30,30))
        X_small.append(small.flatten())

    - Streamlit gives RGB uint8 image [0..255]
    - We convert to float [0..1], then same steps.
    """
    # Convert to float32 [0, 1]
    img_float = image.astype("float32") / 255.0

    # RGB → grayscale (float)
    gray = color.rgb2gray(img_float)

    # Resize to 30x30 (same as training)
    small = transform.resize(gray, (30, 30), anti_aliasing=True)

    # Flatten to 900-dim vector
    features = small.flatten().astype("float32")

    # Shape (1, n_features) for sklearn pipeline
    return features.reshape(1, -1)

def predict_gesture(image: np.ndarray):
    """
    Run model on one RGB image.
    The loaded 'model' is your Pipeline(PCA -> SVC).
    """
    features = preprocess_image(image)
    label = model.predict(features)[0]  # this is your class string: "A", "B", ..., "SPACE", etc.
    # No predict_proba, since SVC(probability=False) by default
    return label

# =========================
# SIDEBAR
# =========================
st.sidebar.header("How to use")
st.sidebar.markdown(
    """
1. Choose **Upload Image** or **Use Camera**  
2. Show a clear hand gesture (A–Z, SPACE, DELETE, NOTHING)  
3. Click **Predict Gesture**  

This app uses an SVM (with PCA) trained on **30×30 grayscale** images.
"""
)

# =========================
# MAIN UI
# =========================
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

# =========================
# PREDICTION BUTTON
# =========================
if uploaded_image is not None:
    if st.button("Predict Gesture"):
        with st.spinner("Predicting..."):
            label = predict_gesture(uploaded_image)

        st.subheader("Prediction")
        st.markdown(f"### 👉 {label}")

        if label in ["SPACE", "DELETE", "NOTHING"]:
            st.info(f"Special command detected: **{label}**")
else:
    st.info("Please upload or capture an image to start.")
