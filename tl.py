import streamlit as st
import numpy as np
import cv2 
from PIL import Image
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# @st.cache()
# def load_model():
#   model=model
#   return model

st.title("Image Classifier - 10 Categories!")
upload = st.sidebar.file_uploader(label='Upload the Image')
if upload is not None:
  file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes, 1)
  opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB)
  img = Image.open(upload)
  st.image(img,caption='Uploaded Image',width=300)
  # Model saved with Keras model.save()
  # Load your trained model
  model = load_model('Cifar.hdf5')
  if st.button("Predict"):
    st.write("Result")
    x=cv2.resize(opencv_image,(32,32))
    x=np.expand_dims(x,axis=0)
    y=model.predict(x)
    y=np.argmax(y.astype(int),axis=1)
    label='airplane automobile bird cat deer dog frog horse ship truck'.split()
    st.title(label[int(y)])   #(y[i]---label[y[i].astype(int)])
