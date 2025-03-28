import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import torch
from utils import get_pothole_model, detection_img

st.title("Pothole Object Detection")

classes = ["Background", "Pothole"]
dirname = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(dirname, os.pardir)
pothole_img_dir = os.path.join(root_dir, "images", "pothole_img")

model = get_pothole_model()
model.eval()

# File uploader widget for uploading images
file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

# Camera input widget for taking a photo
camera_file = st.camera_input("Capture Image")

# Slider widgets for setting confidence and IOU thresholds
c1, c2 = st.columns(2)
with c1:
    conf_threshold = float(st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.02))
with c2:
    iou_threshold = float(st.slider("IOU Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.02))

col1, col2 = st.columns(2)
button = st.button("Detect")

# Process image upload or camera input
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
        idx = np.random.choice(range(4), 1)[0]
        default_img_path = os.path.join(pothole_img_dir, f"{idx}.png")
        img = Image.open(default_img_path)
        img = np.array(img)
        img = cv2.resize(img, (480, 480))

    # Display the images
    with col1:
        st.write(title)
        st.image(img)

    with col2:
        st.write("Pothole Object Detection")
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
        
        # Resize frame for consistency
        frame_resized = cv2.resize(frame, (480, 480))

        # Pothole detection
        detect_img = detection_img(model, frame_resized, classes, conf_threshold, iou_threshold)

        # Display in Streamlit
        stframe.image(detect_img, channels="BGR", use_column_width=True)

        # Stop the live stream when the button is pressed again
        if not st.session_state.get(live_button_key, False):
            cap.release()
            cv2.destroyAllWindows()
        break
