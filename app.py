import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# URL for downloading the model
file_id = "1joBLLGeLlP_dNWQeoMoUb6BjcL3yxvmk"
url = 'https://drive.google.com/uc?export=download&id=1joBLLGeLlP_dNWQeoMoUb6BjcL3yxvmk'
model_path = "trained_plant_disease1_model.keras"

# Check if model exists, if not download it
if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# Ensure the model file exists before loading it
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found, please check the download.")
else:
    # Load the model once
    model = tf.keras.models.load_model(model_path)

    def model_prediction(test_image):
        image = Image.open(test_image)
        image = image.resize((128, 128))  # Resize image
        input_arr = np.array(image)  # Convert to array
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
        predictions = model.predict(input_arr)
        return np.argmax(predictions)  # Return the index of the highest prediction

    # Sidebar setup
    st.sidebar.title("PotatoVision: CNN-Driven Detection of Leaf Diseases for Sustainable Agriculture")
    app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

    # Display image for the developer info
    img = Image.open("Diseases1.jpg")
    st.image(img)

    st.sidebar.write("### About the Developer")
    st.sidebar.write("**Bipin Gundala**")
    st.sidebar.write("I am a software developer with a passion for building machine learning applications.")
    st.sidebar.write("This project is aimed at helping farmers identify plant diseases using deep learning.")
    st.sidebar.write("**Contact Information:**")
    st.sidebar.write("- **Email:** bipin.gundala@gmail.com")
    st.sidebar.write("- **LinkedIn:** [LinkedIn Profile](https://www.linkedin.com/in/bipin-gundala-b54496210/)")

    # Main Page
    if app_mode == "HOME":
        st.markdown("<h1 style='text-align: center;'>PotatoVision: CNN-Driven Detection of Leaf Diseases for Sustainable Agriculture</h1>", unsafe_allow_html=True)

    # Disease Recognition Page
    elif app_mode == "DISEASE RECOGNITION":
        st.header("PotatoVision: CNN-Driven Detection of Leaf Diseases for Sustainable Agriculture")
        test_image = st.file_uploader("Choose an Image:")

        if test_image:
            st.image(test_image, width=400)

        if st.button("Predict") and test_image:
            st.snow()  # Snow effect to show loading
            st.write("Our Prediction")

            # Call model prediction function
            result_index = model_prediction(test_image)

            # Class names
            class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

            # Display the prediction result
            st.success(f"Model is predicting it's a {class_names[result_index]}")
