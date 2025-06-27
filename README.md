# CodeAlpha Object Detection

## Features
- Real-time object detection using YOLOv8 (Ultralytics)
- Works with webcam or video files
- (Optional) Object tracking with Deep SORT

## Setup
```bash
pip install ultralytics opencv-python
# For tracking:
pip install deep_sort_realtime
```

## Usage
- Run detection with webcam:
  ```bash
  python detect.py
  ```
- Run detection with a video file:
  - Edit `detect.py` and set `cv2.VideoCapture('sample_video.mp4')`

- Press `q` to quit the window.
