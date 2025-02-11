import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os

file_id = "1joBLLGeLlP_dNWQeoMoUb6BjcL3yxvmk"
url = 'https://drive.google.com/uc?export=download&id=1joBLLGeLlP_dNWQeoMoUb6BjcL3yxvmk'
model_path = "trained_plant_disease1_model.keras"


if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)
model_path = "trained_plant_disease1_model.keras"

   



def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("PotatoVision: CNN-Driven Detection of Leaf Diseases for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page",["HOME","DISEASE RECOGNITION"])
#app_mode = st.sidebar.selectbox("Select Page",["Home"," ","Disease Recognition"])

# import Image from pillow to open images
from PIL import Image
img = Image.open("Diseases1.jpg")

# display image using streamlit
# width is used to set the width of an image
st.image(img)
st.sidebar.write("### About the Developer")
st.sidebar.write("**Bipin Gundala**")
st.sidebar.write("I am a software developer with a passion for building machine learning applications.")
st.sidebar.write("This project is aimed at helping farmers identify plant diseases using deep learning.")
st.sidebar.write("**Contact Information:**")
st.sidebar.write("- **Email:** bipin.gundala@gmail.com")
st.sidebar.write("- **LinkedIn:** [LinkedIn Profile](https://www.linkedin.com/in/bipin-gundala-b54496210/)")

#Main Page
if(app_mode=="HOME"):
    st.markdown("<h1 style='text-align: center;'>PotatoVision: CNN-Driven Detection of Leaf Diseases for Sustainable Agriculture", unsafe_allow_html=True)
    
#Prediction Page
elif(app_mode=="DISEASE RECOGNITION"):
    st.header("PotatoVision: CNN-Driven Detection of Leaf Diseases for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
