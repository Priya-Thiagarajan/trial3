import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

st.title("Intelligent  Papilledema  Detector  (IPD)")
st.header("By  Priya  Thiagarajan  (21CS007)")

img = st.file_uploader("", type=['png', 'jpg', 'jpeg'])

model_name = 'new_model.h5'
model = load_model(model_name)

pred = model.predict(img)
x = np.argmax(pred)

if x == 0:
  st.success("Normal")
if x == 1:
  st.warning("Pseudopapilledema")
if x == 2:
  st.error("Papilledema")
