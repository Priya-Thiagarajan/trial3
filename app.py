import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
from keras.utils import load_img, img_to_array
from PIL import Image, ImageEnhance

st.title("Intelligent  Papilledema  Detector  (IPD)")
st.header("By  Priya  Thiagarajan  (21CS007)")

files = st.file_uploader("", type=['png', 'jpg', 'jpeg'])

#if (st.button("Submit")): 
  
  #basepath = os.path.dirname(__file__)
  #filename = files.filename
  #filename = filename.replace(" ", "")
  #file_path = os.path.join(basepath, 'uploads', filename)
  #files.save(file_path)

model_name = 'new_model.h5'
model = load_model(model_name)

if (st.button("Submit")):
  #img = image.load_img(file_path, target_size=[240, 240])
  img = Image.open(files)
  img = img_to_array(img)
  #img = img.reshape(240,240)
  img = np.expand_dims(img, axis=0)
  op = model.predict(img)
  x = np.argmax(op)
  
  if x == 0:
    st.success("Normal")
  if x == 2:
    st.warning("Pseudopapilledema")
  if x == 1:
    st.error("Papilledema")
