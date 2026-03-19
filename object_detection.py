"""
Object Detection Module
=======================
Detects cell phones, books, and persons in a video frame using YOLOv11.

Requirements:
    pip install ultralytics opencv-python numpy

Setup:
    Set MODEL_PATH to the location of your yolo11s.pt file.
    Default: looks for yolo11s.pt in the same directory as this file.

Usage:
    import cv2
    from object_detection import detectObject

    frame = cv2.imread("image.jpg")
    labels, annotated_frame, person_count, detected_objects = detectObject(frame)
    # detected_objects -> list of strings: "cell phone", "book", "person"
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------------------------------------------
# MODEL PATH — update this if your .pt file is in a different location
# -----------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo11s.pt")

model = YOLO(MODEL_PATH)

CONFIDENCE_THRESHOLD = 0.5


def detectObject(frame, confidence_threshold=CONFIDENCE_THRESHOLD, resize_width=640):
    """
    Detect objects in a single frame.

    Args:
        frame (np.ndarray): BGR image from OpenCV.
        confidence_threshold (float): Minimum confidence to count a detection.
        resize_width (int): Resize width for faster inference (aspect ratio preserved).

    Returns:
        labels_this_frame (list[tuple]): [(label, confidence_score), ...]
        annotated_frame (np.ndarray): Frame with bounding boxes drawn.
        person_count (int): Number of persons detected.
        detected_objects (list[str]): Detected items of interest — "cell phone", "book", "person".
    """
    labels_this_frame = []
    detected_objects = []
    person_count = 0

    if frame is None or not isinstance(frame, np.ndarray):
        raise ValueError("Invalid frame. Provide a valid numpy array.")

    # Resize for speed while keeping aspect ratio
    height, width = frame.shape[:2]
    if width > resize_width:
        aspect_ratio = height / width
        frame = cv2.resize(frame, (resize_width, int(resize_width * aspect_ratio)))

    try:
        results = model(frame)

        for result in results:
            for box in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, score, class_id = box

                if score > confidence_threshold:
                    label = model.names[int(class_id)]
                    labels_this_frame.append((label, float(score)))

                    if label.lower() == "person":
                        person_count += 1
                        detected_objects.append("person")
                    elif label.lower() == "cell phone":
                        detected_objects.append("cell phone")
                    elif label.lower() == "book":
                        detected_objects.append("book")

                    # Draw bounding box and label
                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2
                    )
                    cv2.putText(
                        frame,
                        f"{label} {score:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

        logging.info(f"Detected: {labels_this_frame}")

    except Exception as e:
        logging.error(f"Detection error: {e}")
        raise

    return labels_this_frame, frame, person_count, detected_objects
