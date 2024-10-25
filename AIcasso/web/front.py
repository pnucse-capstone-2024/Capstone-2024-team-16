import json
import os
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
    response = requests.post("http://localhost:8002/upload/", files={"file": buffer.tobytes()})
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
    # upload_choice = st.radio('upload')

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
            async_processing=True,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
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
    
    st.write("Select the features to transfer:")
    transfer_choice = st.radio("Choose features to transfer:", ("Color", "Shape", "Both"), index=2)
    st.session_state["transfer_choice"] = transfer_choice

    # Display relevant options based on the selection
    if transfer_choice in ["Color", "Both"]:
        if st.checkbox(f"Select from gallery (color)", key="gallery_color"):
            images = os.listdir('input')
            images = [f'input/{img}' for img in images]
            selected_image = st.selectbox("Choose an image for color:", images, key="select_color")
            if selected_image:
                image = Image.open(selected_image)
                st.image(image, caption="Selected Color Reference Image", channels="RGB")
                st.session_state["color_image"] = image

        # if st.checkbox("Upload color image", key="color_upload"):
        #     ref_file = st.file_uploader("Upload a color image:", type=["jpg", "jpeg", "png"], key="color")
        #     if ref_file is not None:
        #         ref_img = Image.open(ref_file)
        #         st.image(ref_img, caption="Color Reference Image", channels="RGB")
        #         st.session_state["color_image"] = ref_img

    if transfer_choice in ["Shape", "Both"]:
        if st.checkbox(f"Select from gallery (shape)", key="gallery_shape"):
            images = os.listdir('input')
            images = [f'input/{img}' for img in images]
            selected_image = st.selectbox("Choose an image for shape:", images, key="select_shape")
            if selected_image:
                image = Image.open(selected_image)
                st.image(image, caption="Selected Shape Reference Image", channels="RGB")
                st.session_state["shape_image"] = image

        # if st.checkbox("Upload shape image", key="shape_upload"):
        #     ref_file = st.file_uploader("Upload a shape image:", type=["jpg", "jpeg", "png"], key="shape")
        #     if ref_file is not None:
        #         ref_img = Image.open(ref_file)
        #         st.image(ref_img, caption="Shape Reference Image", channels="RGB")
        #         st.session_state["shape_image"] = ref_img

# Display the processed image in the right column
with right_column:
    if st.button("Transfer"):
        files = {}
        data = {}

        if "user_image" in st.session_state and st.session_state["user_image"] is not None:
            user_img = st.session_state["user_image"]
            if user_img.mode == "RGBA":
                user_img = user_img.convert("RGB")
            img_bytes = io.BytesIO()
            user_img.save(img_bytes, format="JPEG")
            img_bytes.seek(0)
            files["file"] = img_bytes  # User image file

        if transfer_choice == "Both":
            # Normal behavior: send all images as chosen by the user
            if "color_image" in st.session_state:
                color_img = st.session_state["color_image"]
                color_image_name = None

                if isinstance(color_img, Image.Image):
                    color_image_name = st.session_state.get("select_color", st.session_state.get("color_upload"))
                data["color_image_name"] = os.path.basename(color_image_name) if color_image_name else None

            if "shape_image" in st.session_state:
                shape_img = st.session_state["shape_image"]
                shape_image_name = None

                if isinstance(shape_img, Image.Image):
                    shape_image_name = st.session_state.get("select_shape", st.session_state.get("shape_upload"))
                data["shape_image_name"] = os.path.basename(shape_image_name) if shape_image_name else None

        elif transfer_choice == "Color":
            # Send the color image as selected and shape image as user image
            if "color_image" in st.session_state:
                color_img = st.session_state["color_image"]
                color_image_name = None

                if isinstance(color_img, Image.Image):
                    color_image_name = st.session_state.get("select_color", st.session_state.get("color_upload"))
                data["color_image_name"] = os.path.basename(color_image_name) if color_image_name else None

            data["shape_image_name"] = "user_image.png"  # Pass the user image as shape reference

        elif transfer_choice == "Shape":
            # Send the shape image as selected and color image as user image
            if "shape_image" in st.session_state:
                shape_img = st.session_state["shape_image"]
                shape_image_name = None

                if isinstance(shape_img, Image.Image):
                    shape_image_name = st.session_state.get("select_shape", st.session_state.get("shape_upload"))
                data["shape_image_name"] = os.path.basename(shape_image_name) if shape_image_name else None

            data["color_image_name"] = "user_image.png"  # Pass the user image as color reference

        if files:
            mode = st.session_state["transfer_choice"].lower()  # Convert to lowercase to match expected mode in backend
            data["mode"] = mode

            # Add color and shape image names to the data dictionary
            response = requests.post("http://localhost:8002/upload/", files=files, data=data)
            if response.status_code == 200:
                processed_image = Image.open(io.BytesIO(response.content))
                st.session_state["processed_image"] = processed_image
                st.image(processed_image, caption="Processed Image", channels="RGB")
            else:
                st.error(f"Failed to process image. Status code: {response.status_code}")
        else:
            st.error("No images to process. Please upload or select an image.")
