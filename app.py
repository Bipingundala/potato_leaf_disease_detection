import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# URL for downloading the model
file_id = "1joBLLGeLlP_dNWQeoMoUb6BjcL3yxvmk"
url = 'https://drive.google.com/uc?export=download&id=1joBLLGeLlP_dNWQeoMoUb6BjcL3yxvmk'  # Correct download URL
model_path = "trained_plant_disease1_model.keras"

# Debugging: Check current directory and file list
st.write("Current directory:", os.getcwd())  # Print current directory
st.write("Files in the directory before download:", os.listdir())  # List files before download

# Download model if not present
if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# Debugging: Check if model file is downloaded
st.write("Files in the directory after download:", os.listdir())  # List files after download

# Ensure the model file exists before trying to load it
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found, please check the download.")
else:
    # Load the model
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
