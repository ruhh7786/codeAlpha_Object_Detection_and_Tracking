import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

st.set_page_config(page_title="YOLOv8 Object Detection", layout="centered")
st.title("YOLOv8 Object Detection Web App")

# Custom CSS for attractive UI
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
    }
    .main {
        background-color: #ffffffcc;
        border-radius: 16px;
        padding: 2rem 2rem 1rem 2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    }
    .stButton>button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 8px;
        font-size: 1.1rem;
        padding: 0.5rem 1.5rem;
        margin-top: 1rem;
    }
    .stFileUploader>div>div {
        background: #f0f4fa;
        border: 2px dashed #2a5298;
        border-radius: 10px;
        padding: 1.5rem;
    }
    .stProgress>div>div>div {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    }
    .footer {
        position: fixed;
        left: 0; right: 0; bottom: 0;
        width: 100%;
        background: #2a5298;
        color: #fff;
        text-align: center;
        padding: 0.5rem 0;
        font-size: 1rem;
        z-index: 100;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom header with logo and title
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem;">
        <img src="https://cdn-icons-png.flaticon.com/512/3523/3523887.png" width="60"/>
        <div>
            <h1 style="margin-bottom:0; color:#1e3c72;">YOLOv8 Object Detection</h1>
            <p style="margin-top:0; color:#2a5298; font-size:1.2rem;">Detect and track objects in your videos with AI!</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

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
        stframe.image(rgb_frame, channels="RGB", use_container_width=True)
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

# Add a nice footer
st.markdown(
    """
    <div class="footer">
        Made with ❤️ using Streamlit & YOLOv8 | <a href="https://ultralytics.com/" style="color:#fff;text-decoration:underline;">Ultralytics</a>
    </div>
    """,
    unsafe_allow_html=True
) 