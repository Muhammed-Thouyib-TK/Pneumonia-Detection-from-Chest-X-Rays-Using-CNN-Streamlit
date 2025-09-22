#!/usr/bin/env python
# coding: utf-8

# In[6]:


# importing libraries

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

model = tf.keras.models.load_model('Pneumonia_Prediction.h5')

st.title('Pneumonia Detection - Chest X-Ray') # title
st.write('Upload a Chest X-Ray image to predict whether it show signs of Pneumonia.') # short description

uploaded_file = st.file_uploader("Upload the X-Ray File") # to upload a image

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB') # to open the uploaded image and conert it into RGB format
    st.image(img,caption='Successfully Uploaded the Image✅',use_container_width=True) # to show the image along with the caption

    img = img.resize((32,32)) # resizing
    image_array = image.img_to_array(img)/255.0 # rescaling
    image_array = np.expand_dims(image_array,axis=0) # expanding the dimension due to if the model predict 1 image only

    prediction = model.predict(image_array) # to predict
    confidence = float(prediction[0][0]) # to get the confidence value

    if confidence>0.5: # print output
        st.error(f'Prediction : Pneumonia , Confidence : {round(confidence*100,2)}')
    else:
        st.success(f'Prediction : Normal , Confidence : {round(confidence*100,2)}')


# In[ ]:




