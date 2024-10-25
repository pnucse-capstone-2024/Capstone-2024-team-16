import av
import io
import cv2
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import requests
import numpy as np
from PIL import Image
from ultralytics import YOLO

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(layout="wide")

def model_load():
    return YOLO(model="model/yolov8n-face.onnx", task="detect", verbose=False)

DETECTOR = model_load()

def send_image_to_server(image):
    _, buffer = cv2.imencode('.jpg', image)
    response = requests.post("http://localhost:8000/upload/", files={"file": buffer.tobytes()})
    return response.json()

class MyVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.captured_image = None
        self.flip = False

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        if self.flip:
            image = cv2.flip(image, 1)
        self.captured_image = image
        return av.VideoFrame.from_ndarray(image, format="bgr24")

st.title("Image Upload and Webcam Processing")

# Layout with two columns
left_column, right_column = st.columns(2)

# Initialize session state for processed file and last frame
if "processed_image" not in st.session_state:
    st.session_state["processed_image"] = None

if "user_image" not in st.session_state:
    st.session_state["user_image"] = None

# Left column for image upload or webcam
with left_column:
    upload_choice = st.radio("Choose an option:", ("Upload an image", "Use webcam"))

    if upload_choice == "Upload an image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, channels="RGB")
            st.session_state["user_image"] = img

    elif upload_choice == "Use webcam":
        flip = st.checkbox("Flip", value=False)

        ctx = webrtc_streamer(
            key="example",
            video_processor_factory=MyVideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

        if ctx.video_transformer:
            ctx.video_transformer.flip = flip
            if st.button("Capture Image"):
                img = ctx.video_transformer.captured_image
                if img is not None:
                    st.image(img, channels="BGR")
                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    st.session_state["user_image"] = img
                else:
                    st.warning("No image captured yet.")

# Display the processed image in the right column
with right_column:
    if st.button("Transfer"):
        if st.session_state["user_image"] is not None:
            img = st.session_state["user_image"]
            if img.mode == "RGBA":
                img = img.convert("RGB")

            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG")
            img_bytes.seek(0)

            response = requests.post(
                "http://localhost:8000/upload/",
                files={"file": img_bytes}
            )

            if response.status_code == 200:
                st.session_state["processed_image"] = Image.open(io.BytesIO(response.content))

        else:
            st.warning("There is no uploaded image.")

    if st.session_state["processed_image"] is not None:
        st.image(st.session_state["processed_image"], channels="RGB")
    else:
        st.write("No processed image yet.")
