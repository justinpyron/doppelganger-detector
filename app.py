import numpy as np
import streamlit as st
from PIL import Image

from retriever import Retriever

WHAT_IS_THIS_APP = """
This app uses a computer vision model to identify your celebrity doppelganger 💫

#### `Data`
I start with this [list of 1000 celebrities from IMDB](https://www.imdb.com/list/ls058011111/).
For each celebrity, I scraped 10 images from Google Images.

#### `Model`
I use a [Hugging Face Vision Transformer model](https://huggingface.co/trpakov/vit-face-expression) trained to detect facial expressions.

The model outputs an embedding vector that captures visual information from an input image.
The original model outputs a 7-dimensional vector of logits (corresponding to 7 facial expressions).
I modify the final linear layer of the network to output vectors of dimension 128.


#### `Training`
I fine-tune the model using a triplet loss function.
Each triplet consists of three images: anchor, positive, negative.
The triplet loss function trains the model to embed the anchor and positive images close together and the anchor and negative images far apart.

I fine-fune the model in two phases:
1. Only update weights of the final linear layer (which was freshly initialized because it was new)
2. Update weights of all layers

#### `Source code`
See 👉 [GitHub](https://github.com/justinpyron/doppelganger-detector)

#### `Next steps`
The model is a work in progress 🙃

It has weaknesses that I'd like to improve.
For instance, the current model mostly matches to the celebrity image with the most similar clothing, hair style, and background.
I want to explore different modeling and data processing techniques to fix this issue.
"""


@st.cache_resource
def load_retriever():
    return Retriever()


st.set_page_config(page_title="Doppelganger Detector", layout="centered", page_icon="🥸")
doppelganger_retriever = load_retriever()
st.title("Doppelganger Detector 🥸")
with st.expander("What is this app?"):
    st.markdown(WHAT_IS_THIS_APP)

ready = False
input_type_options = ["Camera", "Upload"]
input_type = st.segmented_control(
    label="input_type",
    options=input_type_options,
    label_visibility="hidden",
)
if input_type == input_type_options[0]:
    enable_camera = st.checkbox("Enable camera")
    st.write("ℹ️ Your photo is deleted when you close the page")
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
