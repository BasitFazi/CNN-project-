import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from PIL import Image

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Load trained model
@st.cache_resource
def load_trained_model():
    model = load_model("hand_digit_recognition.h5")  # Ensure model is in the working directory
    return model

model = load_trained_model()

# Streamlit UI
st.title("✋ Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9) to recognize it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img_resized = image.resize((28, 28))
    img_array = np.array(img_resized) / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28)  # Reshape for model
    
    # Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    # Display results
    st.subheader("Prediction Result")
    st.write(f"**Predicted Digit:** {predicted_digit}")
    st.write(f"**Confidence:** {confidence:.2f}%")
