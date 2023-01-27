import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
from keras.utils import load_img, img_to_array

st.title("Intelligent  Papilledema  Detector  (IPD)")
st.header("By  Priya  Thiagarajan  (21CS007)")

img = st.file_uploader("", type=['png', 'jpg', 'jpeg'])

model_name = 'new_model.h5'
model = load_model(model_name)

img = img_to_array(img)
img = numpy.expand_dims(img, axis=0)
op = model.predict(img)
x = np.argmax(op)

if x == 0:
  st.success("Normal")
if x == 1:
  st.warning("Pseudopapilledema")
if x == 2:
  st.error("Papilledema")
