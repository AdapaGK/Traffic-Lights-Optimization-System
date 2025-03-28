from cmath import rect
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import torch
from utils import classify_img, get_density_model

st.title("Traffic Density Classification")

classes = ['Empty', 'High', 'Low', 'Medium', 'Traffic Jam']
model = get_density_model()
model.eval()

# Directory paths for default images
dirname = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(dirname, os.pardir)
density_img_dir = os.path.join(root_dir, "images", "density_img")

# File uploader widget for uploading images
file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

# Camera input widget for taking a photo
camera_file = st.camera_input("Capture Image")

# Slider for adjusting the confidence threshold (optional)
confidence_slider = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# Button to start detection (image or live)
button = st.button("Detect")

if button:
    if file is not None:
        # Image uploaded
        title = "Uploaded Image"
        img = Image.open(file)
        img = np.array(img)
        img = cv2.resize(img, (480, 480))
        label, probability = classify_img(model, img)
    elif camera_file is not None:
        # Image taken from camera
        title = "Captured Image"
        img = Image.open(camera_file)
        img = np.array(img)
        img = cv2.resize(img, (480, 480))
        label, probability = classify_img(model, img)
    else:
        # Default image if no file or camera input
        title = "Default Image"
        idx = np.random.choice(range(5), 1)[0]
        default_img_path = os.path.join(density_img_dir, f"{idx}.jpg")
        img = Image.open(default_img_path)
        img = np.array(img)
        img = cv2.resize(img, (480, 480))
        label, probability = classify_img(model, img)

    # Display the results
    st.success(f"Predicted Class is {classes[label]} with probability {probability:.4f}")
    st.image(img, caption=title)

# Real-time live detection using OpenCV and Streamlit
live_button_key = "live_button"  # Unique key for the button
if st.button("Start Live Detection", key=live_button_key):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)  # Capture from webcam

    while True:
        ret, frame = cap.read()
        if not rect :
            break

        # Resize the frame for consistency
        frame_resized = cv2.resize(frame, (480, 480))

        # Traffic density classification
        label, probability = classify_img(model, frame_resized)

        # Show the classification result on the frame
        result_text = f"{classes[label]}: {probability:.4f}"
        cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the frame with the result
        stframe.image(frame, channels="BGR", use_column_width=True)

        # Optionally, stop the live stream if the button is pressed again
        if not st.session :
            n_state.get(live_button_key, False) # type: ignore
        cap.release()
        cv2.destroyAllWindows()
        break
