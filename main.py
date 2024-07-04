import streamlit as st
import cv2
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import os
import urllib.request

# Ensure model file exists
model_path = 'yolov8m.pt'
if not os.path.exists(model_path):
    st.warning("Model file not found. Downloading...")
    # Download model from a URL (you need to host it somewhere)
    model_url = 'https://path/to/your/yolov8m.pt'
    urllib.request.urlretrieve(model_url, model_path)
    st.success("Model downloaded successfully!")

# Load a larger YOLOv8 model for better accuracy
model = YOLO(model_path)  # You can use 'yolov8m.pt', 'yolov8l.pt', etc., for more accuracy

st.title("YOLOv8 Object Detection from Camera Feed")

# Initialize session state for camera
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

# Function to start the camera
def start_camera():
    st.session_state.camera_active = True

# Function to stop the camera
def stop_camera():
    st.session_state.camera_active = False

# Start and Stop buttons
if not st.session_state.camera_active:
    if st.button('Start Camera'):
        start_camera()
else:
    if st.button('Stop Camera'):
        stop_camera()

# Video capture and YOLO object detection
if st.session_state.camera_active:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    with tf.device('/device:GPU:0'):  # Ensuring we use the GPU if available
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video")
                break

            # Perform inference on the frame
            results = model(frame)

            # Extract detection results
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                scores = result.boxes.conf.cpu().numpy()  # Confidence scores
                classes = result.boxes.cls.cpu().numpy()  # Class IDs

                # Draw bounding boxes and labels on the frame
                for i in range(len(boxes)):
                    box = boxes[i]
                    score = scores[i]
                    class_id = int(classes[i])
                    label = model.names[class_id]

                    if score > 0.5:  # Adjusted threshold for more confident detections
                        # Extract box coordinates
                        start_x, start_y, end_x, end_y = map(int, box[:4])

                        # Draw the bounding box
                        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

                        # Draw the label and score
                        label_text = f"{label}: {score:.2f}"
                        cv2.putText(frame, label_text, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
            stframe.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()
