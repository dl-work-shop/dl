import streamlit as st #type: ignore
import numpy as np
from keras_preprocessing.image import ImageDataGenerator #type:ignore
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout #type:ignore
from tensorflow.keras.models import Model, load_model #type:ignore
from glob import glob
import os
import sys
sys.path.append('../src')
from get_data import read_params,get_data #type:ignore
import argparse
import matplotlib.pyplot as plt 
from keras.applications.vgg16 import VGG16 #type:ignore
import tensorflow as tf #type: ignore
import mlflow #type: ignore
from urllib.parse import urlparse
import mlflow.keras #type: ignore
from PIL import Image
import cv2 #type: ignore
import base64

st.set_page_config(page_title="Deep Learning", page_icon="🧠", layout="wide")
st.markdown("""
    <style>
    .reportview-container { background: #f5f5f5; }
    .stButton > button { width: 100%; border-radius: 10px; }
    .stImage { text-align: center; }
    </style>
    """, unsafe_allow_html=True)


model = load_model('../models/trained.h5')

#Class Labels

classes = ['1','1','1','1','glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']

def preprocess_image(image):
    image = tf.image.resize(image, (255, 255))
    image = np.array(image)  # Convert to numpy array
    image = image.copy()  # Create a writable copy
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image /= 255.0  # Normalize
    return image

def generate_gradcam(image, model):
    """Generate GradCAM Visualization"""
    #Dummy implementation (replace with actual GRAD_CAM logic)

    img_array = np.array(image)
    heatmap = np.uint8(255 * np.random.rand(*img_array.shape[:2]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return Image.fromarray(heatmap)

def download_report(pred_class,confidence):
    """Generate and return a downloadable report"""

    report_text = f""" 
    Brain Tumor Classification Report
    -------------------------------
    Predicted : {pred_class}
    Confidence: {confidence:.2f}%
    """

    b64 = base64.b64encode(report_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="report.txt">Download Report</a>'
    return href

# Sidebar
st.sidebar.title("Upload MRI Scan")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Scan', use_column_width=True)

    if st.button("Classify Image"):
        input_tensor = preprocess_image(image)
        output = model.predict(input_tensor)
        
        # Debug prints
        st.write("Model output:", output)
        
        pred_idx = np.argmax(output, axis=1)[0]
        
        # Debug prints
        st.write("Predicted index:", pred_idx)
        
        if pred_idx < len(classes):
            confidence = output[0][pred_idx] * 100
            pred_class = classes[pred_idx]

            # Display Result
            col1, col2 = st.columns([2, 1])
            with col1:
                st.success(f"Prediction: {pred_class}")
                st.info(f"Confidence: {confidence:.2f}%")

            with col2:
                gradcam_images = generate_gradcam(image, model)
                st.image(gradcam_images, caption='GradCAM Visualization', use_column_width=True)

            # Download Report
            st.markdown(download_report(pred_class, confidence), unsafe_allow_html=True)
        else:
            st.error("Prediction index out of range. Please check the model output.")