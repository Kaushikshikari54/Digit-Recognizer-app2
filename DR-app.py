import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib
from sklearn.datasets import load_digits

st.set_page_config(page_title="Digit Recognizer", page_icon="🔢")

model = joblib.load('digit_classifier.pkl')
st.title("🔢 Digit Recognizer App (No TensorFlow)")
st.write("Upload a grayscale **8x8** PNG image of a digit (0–9)")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Process uploaded image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)
    image = image.resize((8, 8))  # Resize to 8x8 to match the dataset
    img_array = np.array(image)

    st.image(image, caption="Processed Image", width=150)

    # Normalize and flatten like sklearn expects
    img_array = (16 - img_array / 16).flatten().reshape(1, -1)

    prediction = model.predict(img_array)
    st.success(f"### Predicted Digit: {prediction[0]}")
