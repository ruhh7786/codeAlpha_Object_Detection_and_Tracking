import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

st.set_page_config(page_title="YOLOv8 Object Detection", layout="centered")
st.title("YOLOv8 Object Detection Web App")

st.write("Upload a video file to run object detection. Optionally, enable tracking (coming soon).")

# Sidebar options
use_tracking = st.sidebar.checkbox("Enable Object Tracking (Deep SORT)", value=False, disabled=True)

# File uploader
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Load YOLOv8 model
    model = YOLO('yolov8m.pt')

    # Video capture
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = tempfile.mktemp(suffix='.mp4')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    stframe = st.empty()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)

    for idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        # Run YOLOv8 detection
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        # Convert BGR to RGB for display
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB", use_column_width=True)
        progress.progress(min((idx+1)/frame_count, 1.0))
    cap.release()
    out.release()
    st.success("Processing complete!")

    with open(out_path, "rb") as f:
        st.download_button("Download Processed Video", f, file_name="output_yolo.mp4", mime="video/mp4")

    # Clean up temp files
    os.remove(video_path)
    os.remove(out_path)
else:
    st.info("Please upload a video file to begin.") 