
import streamlit as st 
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model=tf.keras.models.load_model("trained_plant_disease1_model.keras")
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    predictions=model.predict(input_arr)
    return np.argmax(predictions)
st.sidebar.title("PotatoVision: CNN-Driven Detection of Leaf Diseases for Sustainable Agriculture")
app_mode=st.sidebar.selectbox('select page',['Home','Disease Recognition'])

st.sidebar.write("### About the Developer")
st.sidebar.write("**Bipin Gundala**")
st.sidebar.write("I am a software developer with a passion for building machine learning applications.")
st.sidebar.write("This project is aimed at helping farmers identify plant diseases using deep learning.")
st.sidebar.write("**Contact Information:**")
st.sidebar.write("- **Email:** bipin.gundala@gmail.com")
st.sidebar.write("- **LinkedIn:** [LinkedIn Profile](https://www.linkedin.com/in/bipin-gundala-b54496210/)")

from PIL import Image
img=Image.open('Diseases1.jpg')
st.image(img)

if(app_mode == 'Home'):
    st.markdown("<h1 style='text-align:center;'> PotatoVision: CNN-Driven Detection of Leaf Diseases for Sustainable Agriculture",unsafe_allow_html=True)
elif(app_mode == 'Disease Recognition'):
    st.header('PotatoVision: CNN-Driven Detection of Leaf Diseases for Sustainable Agriculture')
test_image=st.file_uploader("choose an Image")
if(st.button('Show image')):
    st.image(test_image,width=4,use_column_width=True)
if(st.button('Predict')):
    st.snow()
    st.write('Our Prediction')
    result_index=model_prediction(test_image)
    class_name=['Potato__Early_bright','Potato__Late_blight','Potato__healthy']
    st.success('Model is Predicting its a {}'.format(class_name[result_index]))