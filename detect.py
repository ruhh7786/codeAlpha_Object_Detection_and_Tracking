import cv2
from ultralytics import YOLO
import sys

# Load YOLOv8 model (nano version for speed)
model = YOLO('yolov8m.pt')  # or 'yolov8l.pt', 'yolov8x.pt'

# Use 0 for webcam, or replace with 'sample_video.mp4' for a video file
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run detection
    results = model(frame)
    annotated_frame = results[0].plot()  # Draws boxes & labels

    # Display
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 27 is ESC
        break

cap.release()
cv2.destroyAllWindows()
sys.exit(0)
