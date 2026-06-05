import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import streamlit as st
import numpy as np
from PIL import Image
from predict import load_model, predict
from gradcam import GradCAM

st.set_page_config(page_title="Pneumonia Detector", page_icon="??", layout="centered")
st.title("?? Pneumonia Detection from Chest X-Rays")
st.markdown("Upload a chest X-ray and the AI will detect pneumonia and highlight where it found the abnormality.")
st.markdown("---")

@st.cache_resource
def get_model():
    model = load_model()
    gradcam = GradCAM(model)
    return model, gradcam

model, gradcam = get_model()
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_path = "temp_xray.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    result = predict(temp_path, model)
    cam = gradcam.generate(result["tensor"])
    overlay = gradcam.overlay(cam, result["original_image"])
    prediction = result["prediction"]
    confidence = result["confidence"]
    probs = result["probabilities"]
    st.markdown("---")
    if prediction == "PNEUMONIA":
        st.markdown(f"## ?? {prediction} DETECTED")
        st.error(f"Confidence: {confidence}%")
    else:
        st.markdown(f"## ?? {prediction}")
        st.success(f"Confidence: {confidence}%")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original X-Ray")
        st.image(result["original_image"].resize((224, 224)), use_container_width=True)
    with col2:
        st.subheader("Grad-CAM Heatmap")
        st.image(overlay, use_container_width=True)
    st.markdown("---")
    st.subheader("Probabilities")
    st.metric("NORMAL", f"{probs['NORMAL']}%")
    st.progress(int(probs["NORMAL"]))
    st.metric("PNEUMONIA", f"{probs['PNEUMONIA']}%")
    st.progress(int(probs["PNEUMONIA"]))
    os.remove(temp_path)
