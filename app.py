import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

@st.cache_resource
def load_pneumonia_model():
    model_path = 'pneumonia_model_custom.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        return None

def predict_and_display(img, model):
    st.image(img, caption='Selected X-ray', use_container_width=True)
    st.write("Processing X-ray and analyzing for pneumonia...")
    
    img = img.convert('RGB')
    img = img.resize((180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    if model is not None:
        with st.spinner("Classifying..."):
            prediction = model.predict(img_array)
            confidence_normal = prediction[0][0] * 100
            confidence_pneumonia = prediction[0][1] * 100
            
            if prediction[0][1] > prediction[0][0]:
                st.error(f"**Prediction: PNEUMONIA DETECTED!** (Confidence: {confidence_pneumonia:.2f}%)")
                st.write("The model identified patterns typical of Pneumonia. Please consult a healthcare professional.")
            else:
                st.success(f"**Prediction: NORMAL** (Confidence: {confidence_normal:.2f}%)")
                st.write("The X-ray appears normal, without evident signs of Pneumonia.")

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="centered")
st.title("🫁 Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image or choose a sample to detect whether it shows signs of Pneumonia or if it is Normal. This model uses a custom CNN architecture.")

model = load_pneumonia_model()

if model is None:
    st.warning("Model not found. Please run `train_model.py` first to train the model and generate `pneumonia_model_custom.keras`.")

st.markdown("### Test with Sample Images")
st.write("Click on any sample image below to see how the model predicts it.")

# Hardcoding sample paths from the test directory
sample_images = [
    {"path": os.path.join("chest_xray", "test", "NORMAL", "NORMAL-1049278-0001.jpeg"), "label": "Normal Sample"},
    {"path": os.path.join("chest_xray", "test", "NORMAL", "NORMAL-11419-0001.jpeg"), "label": "Normal Sample"},
    {"path": os.path.join("chest_xray", "test", "PNEUMONIA", "BACTERIA-1135262-0001.jpeg"), "label": "Pneumonia Sample"},
    {"path": os.path.join("chest_xray", "test", "PNEUMONIA", "VIRUS-1056329-0001.jpeg"), "label": "Pneumonia Sample"}
]

cols = st.columns(4)
for idx, col in enumerate(cols):
    sample = sample_images[idx]
    if os.path.exists(sample["path"]):
        img_thumb = Image.open(sample["path"])
        # Display thumbnail
        col.image(img_thumb, use_container_width=True)
        if col.button(f"Test {sample['label']}", key=f"btn_{idx}"):
            st.markdown("---")
            st.subheader("Sample Image Prediction")
            predict_and_display(img_thumb, model)

st.markdown("---")
st.markdown("### Or Upload Your Own Image")

uploaded_file = st.file_uploader("Choose a Chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.markdown("---")
    st.subheader("Uploaded Image Prediction")
    img = Image.open(uploaded_file)
    predict_and_display(img, model)
