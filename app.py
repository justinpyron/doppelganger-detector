import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Doppelganger Detector", layout="centered", page_icon="🥸")

st.title("Doppelganger Detector 🥸")

input_type_options = ["Camera", "Upload"]
input_type = st.segmented_control(
    label="input_type",
    options=input_type_options,
    label_visibility="hidden",
)

if input_type == input_type_options[0]:
    enable_camera = st.checkbox("Enable camera")
    if enable_camera:
        photo_camera = st.camera_input("Take a photo", disabled=not enable_camera)
        if photo_camera:
            with Image.open(photo_camera) as img:
                image = np.array(img)
            st.write(f"Photo converted to numpy array of shape {image.shape}")
if input_type == input_type_options[1]:
    photo_uploaded = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])
    if photo_uploaded:
        with Image.open(photo_uploaded) as img:
            x = np.array(img)
        st.write(f"Uploaded photo converted to numpy array of shape {x.shape}")
