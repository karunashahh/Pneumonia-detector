
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

st.title("Pneumonia Detector — Chest X-Ray Classifier")
st.caption("MobileNetV2, fine-tuned | Pneumonia screening classifier")

IMG_SIZE = (160, 160)
THRESHOLD = 0.5

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pneumonia_mobilenetv2_clean_split.keras")

model = load_model()

uploaded = st.file_uploader("Upload a chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB").resize(IMG_SIZE)
    st.image(img, caption="Uploaded X-ray", width=300)

    arr = np.expand_dims(np.array(img), axis=0)
    arr = preprocess_input(arr)

    prob = model.predict(arr, verbose=0)[0][0]
    label = "PNEUMONIA" if prob >= THRESHOLD else "NORMAL"

    st.subheader(f"Prediction: {label}")
    st.write(f"Model confidence score: {prob:.2f}  (flag threshold: {THRESHOLD})")

    if label == "PNEUMONIA":
        st.warning("This is a screening tool, not a diagnosis. Consult a radiologist.")
