import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

st.title("Intelligent  Papilledema  Detector  (IPD)")
st.header("By  Priya  Thiagarajan  (21CS007)")

img = st.file_uploader("Choose a File")

model_name = 'new_model.h5'
model = load_model(model_name)

pred = model.pred(img)
x = np.argmax(pred)

if x == 0:
  st.success("Normal")
if x == 1:
  st.warning("Pseudopapilledema")
if x == 2:
  st.error("Papilledema")
