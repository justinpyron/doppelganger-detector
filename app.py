import numpy as np
import streamlit as st
from PIL import Image

from retriever import Retriever


@st.cache_resource
def load_retriever():
    return Retriever()


st.set_page_config(page_title="Doppelganger Detector", layout="centered", page_icon="🥸")
st.title("Doppelganger Detector 🥸")
doppelganger_retriever = load_retriever()

ready = False
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
            ready = True
if input_type == input_type_options[1]:
    photo_uploaded = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])
    if photo_uploaded:
        with Image.open(photo_uploaded) as img:
            image = np.array(img)
        st.image(photo_uploaded, width=200)
        ready = True

submitted = st.button("Submit", use_container_width=True, type="primary")
if submitted and ready:
    result = doppelganger_retriever.find(image, k_retrieve=50, k_return=3)
    st.header("Top matches")
    for i, r in enumerate(result):
        name_processed = " ".join([x.capitalize() for x in r.split("_")])
        st.markdown(f"{i+1}. `{name_processed}`")
