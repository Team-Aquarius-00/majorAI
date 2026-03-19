"""
Facial Detection Module
=======================
Detects faces and facial landmarks in a video frame using MediaPipe.
Alerts when multiple faces are detected.

Requirements:
    pip install mediapipe opencv-python numpy

Usage:
    import cv2
    from facial_detection import detectFace

    frame = cv2.imread("image.jpg")
    face_count, annotated_frame = detectFace(frame)

    if face_count == 0:
        print("No face detected")
    elif face_count > 1:
        print("Multiple faces — possible cheating!")
    else:
        print("Single face — OK")
"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe models (loaded once at import time)
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)


def detectFace(frame):
    """
    Detect faces and draw landmarks on the frame.

    Args:
        frame (np.ndarray): BGR image from OpenCV.

    Returns:
        face_count (int): Number of faces detected.
        annotated_frame (np.ndarray): Frame with bounding boxes and mesh drawn.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_count = 0
    annotated_frame = frame.copy()

    # Detect faces
    detection_results = face_detection.process(rgb_frame)
    if detection_results.detections:
        face_count = len(detection_results.detections)
        for detection in detection_results.detections:
            mp_drawing.draw_detection(annotated_frame, detection)

    # Alert overlay for multiple faces
    if face_count > 1:
        cv2.putText(
            annotated_frame,
            "Alert: Multiple Faces Detected!",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    # Draw facial landmark mesh
    mesh_results = face_mesh.process(rgb_frame)
    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=1
                ),
            )

    return face_count, annotated_frame
