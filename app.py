import streamlit as st
import numpy as np
import joblib
from pathlib import Path
from skimage import color, transform
from PIL import Image
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
st.write("Model: SVM  Classes: A–Z, SPACE, DELETE, NOTHING")

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
    Ensure svm_final.pkl exists.

    If svm_final.pkl is missing but svm_final.zip is present:
      - open the zip
      - find the first .pkl inside
      - extract it
      - rename it to svm_final.pkl (if needed)
    """
    if MODEL_PKL.exists():
        return

    if not MODEL_ZIP.exists():
        st.error(
            "❌ Model not found.\n\n"
            "I looked for 'svm_final.pkl' and 'svm_final.zip' in the app folder, "
            "but neither exists.\n\n"
            "Please make sure you uploaded **svm_final.zip** to the repo."
        )
        st.stop()

    try:
        with zipfile.ZipFile(MODEL_ZIP, "r") as zf:
            # Find any .pkl file inside the zip
            pkl_names = [n for n in zf.namelist() if n.lower().endswith(".pkl")]
            if not pkl_names:
                st.error(
                    "❌ No .pkl file found inside svm_final.zip.\n\n"
                    "Please check the zip content. It must contain your SVM model (.pkl)."
                )
                st.stop()

            # Take the first .pkl file
            member = pkl_names[0]
            extracted_path_str = zf.extract(member)  # filepath created on disk
            extracted_path = Path(extracted_path_str)

            # If the extracted file isn't named svm_final.pkl, rename it
            if extracted_path.name != MODEL_PKL.name:
                extracted_path.rename(MODEL_PKL)

    except Exception as e:
        st.error(f"❌ Failed to extract model from svm_final.zip: {e}")
        st.stop()


@st.cache_resource
def load_model(model_path: str):
    # Ensure .pkl exists (extract from zip if needed)
    ensure_model_unzipped()

    path = Path(model_path)
    if not path.exists():
        st.error(f"❌ Model file not found after extraction: {path.resolve()}")
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

    - image: RGB uint8 [0..255]
    - convert to float [0..1], then same steps.
    """
    img_float = image.astype("float32") / 255.0
    gray = color.rgb2gray(img_float)
    small = transform.resize(gray, (30, 30), anti_aliasing=True)
    features = small.flatten().astype("float32")
    return features.reshape(1, -1)


def predict_gesture(image: np.ndarray):
    """
    Run model on one RGB image.
    The loaded 'model' is your Pipeline(PCA -> SVC).
    """
    features = preprocess_image(image)
    label = model.predict(features)[0]  # "A", "B", ..., "SPACE", etc.
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
        img = Image.open(file).convert("RGB")
        img_np = np.array(img)
        uploaded_image = img_np
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

else:  # Use Camera
    camera_img = st.camera_input("Capture a hand gesture")
    if camera_img is not None:
        img = Image.open(camera_img).convert("RGB")
        img_np = np.array(img)
        uploaded_image = img_np
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
