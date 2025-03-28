import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import torch
from utils import get_category_model, detection_img

st.title("Vehicle Category Detection")

classes = ["background", "Auto", "Bus", "Car", "LCV", "Motorcycle", "Truck", "Tractor", "Multi-Axle"]
model = get_category_model()
model.eval()

# Directory paths for default images
dirname = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(dirname, os.pardir)
category_img_dir = os.path.join(root_dir, "images", "category_img")

# File uploader widget for uploading images
file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

# Camera input widget for taking a photo
camera_file = st.camera_input("Capture Image")

# Slider widgets for setting confidence and IOU thresholds
c1, c2 = st.columns(2)
with c1:
    conf_threshold = float(st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.02))
with c2:
    iou_threshold = float(st.slider("IOU Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.02))

# Button to start detection (image or live)
button = st.button("Detect")

if button:
    if file is not None:
        # Image uploaded
        title = "Uploaded Image"
        img = Image.open(file)
        img = np.array(img)
        img = cv2.resize(img, (480, 480))
    elif camera_file is not None:
        # Image taken from camera
        title = "Captured Image"
        img = Image.open(camera_file)
        img = np.array(img)
        img = cv2.resize(img, (480, 480))
    else:
        # Default image if no file or camera input
        title = "Default Image"
        idx = np.random.choice(range(7), 1)[0]
        default_img_path = os.path.join(category_img_dir, f"{idx}.jpg")
        img = Image.open(default_img_path)
        img = np.array(img)
        img = cv2.resize(img, (480, 480))

    # Display the images
    with st.columns(2)[0]:
        st.write(title)
        st.image(img)

    with st.columns(2)[1]:
        st.write("Vehicle Category Detection")
        detect_img = detection_img(model, img, classes, conf_threshold, iou_threshold)
        st.image(detect_img)

# Real-time live detection using OpenCV and Streamlit
live_button_key = "live_button"  # Unique key for the button
if st.button("Start Live Detection", key=live_button_key):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)  # Capture from webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for consistency
        frame_resized = cv2.resize(frame, (480, 480))

        # Vehicle category classification
        detect_img = detection_img(model, frame_resized, classes, conf_threshold, iou_threshold)

        # Display the frame with the result
        stframe.image(detect_img, channels="BGR", use_column_width=True)

        # Optionally, stop the live stream if the button is pressed again
        if not st.session_state.get(live_button_key, False):
            cap.release()
            cv2.destroyAllWindows()
            break
