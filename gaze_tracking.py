"""
Gaze Tracking Module
====================
Tracks the user's gaze direction (left / right / center) using MediaPipe Face Mesh.

Requirements:
    pip install mediapipe opencv-python numpy

Usage:
    import cv2
    from gaze_tracking import gaze_tracking

    frame = cv2.imread("image.jpg")
    result = gaze_tracking(frame)
    print(result["gaze"])  # "left", "right", or "center"

Integration tip:
    Call this on every webcam frame. If "gaze" != "center" for N consecutive
    frames, flag it as a suspicious event (avoids false positives from blinking).
"""

import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh (loaded once at import time)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)


def gaze_tracking(frame):
    """
    Determine gaze direction from a video frame.

    Args:
        frame (np.ndarray): BGR image from OpenCV.

    Returns:
        dict: {"gaze": "left" | "right" | "center"}
              Returns {"gaze": "center"} if no face is detected (safe default).
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # Key eye landmark indices (MediaPipe Face Mesh)
            left_eye = [
                landmarks.landmark[33],
                landmarks.landmark[159],
            ]  # Left eye corners
            right_eye = [
                landmarks.landmark[362],
                landmarks.landmark[386],
            ]  # Right eye corners

            left_eye_center = np.mean([(p.x, p.y) for p in left_eye], axis=0)
            right_eye_center = np.mean([(p.x, p.y) for p in right_eye], axis=0)

            gaze_direction = "center"
            if left_eye_center[0] < 0.4:  # Looking left
                gaze_direction = "left"
            elif right_eye_center[0] > 0.6:  # Looking right
                gaze_direction = "right"

            return {"gaze": gaze_direction}

    # No face found — default to center to avoid false alerts
    return {"gaze": "center"}
