"""
Face Verification Module
========================
Encodes a face from an image and compares it against a stored encoding.
Used for identity verification at login — ensures the person at the camera
matches the registered user.

Requirements:
    pip install face-recognition numpy
    # Also requires dlib:
    pip install dlib-19.22.99-cp310-cp310-win_amd64.whl  (Python 3.10, Windows)

⚠️  Important:
    - Only works with Python 3.10 on Windows (due to dlib .whl constraint).
      On Linux/Mac, install dlib normally: pip install dlib
    - face_recognition works best with clear, front-facing, well-lit photos.
    - Store face_encoding as a list (JSON-serializable) in your database.
      On retrieval, convert back to np.array before comparing.

Usage:
    import cv2, numpy as np
    from face_verification import get_face_encoding, match_face_encodings

    # --- At registration ---
    image = cv2.imread("user_photo.jpg")
    encoding = get_face_encoding(image)        # Returns np.ndarray or None
    if encoding is None:
        print("No face found — ask user to retake photo")
    else:
        store_in_db(encoding.tolist())         # Save as JSON list

    # --- At login ---
    live_frame = cv2.imread("live_capture.jpg")
    live_encoding = get_face_encoding(live_frame)
    stored_encoding = np.array(load_from_db()) # Convert back from list

    if live_encoding is not None:
        is_match = match_face_encodings(live_encoding, stored_encoding)
        print("Access granted" if is_match else "Face does not match")
"""

import numpy as np
import face_recognition


def get_face_encoding(image):
    """
    Extract the 128-dimensional face encoding from an image.

    Args:
        image (np.ndarray): BGR image (from OpenCV) or RGB image.
                            face_recognition works with RGB — if passing
                            an OpenCV BGR frame, it's handled internally.

    Returns:
        np.ndarray: 128-d face encoding array, or None if no face is found.
    """
    # face_recognition expects RGB
    if image.shape[2] == 3:
        rgb_image = image[:, :, ::-1]  # Convert BGR -> RGB (OpenCV default is BGR)
    else:
        rgb_image = image

    face_locations = face_recognition.face_locations(rgb_image)
    if not face_locations:
        return None

    encodings = face_recognition.face_encodings(rgb_image, face_locations)
    return encodings[0] if encodings else None


def match_face_encodings(live_encoding, stored_encoding, tolerance=0.6):
    """
    Compare a live face encoding with a stored encoding.

    Args:
        live_encoding (np.ndarray): Encoding from the current webcam frame.
        stored_encoding (np.ndarray): Encoding loaded from database.
        tolerance (float): Lower = stricter matching. Default 0.6 is recommended.
                           Use 0.5 for higher security.

    Returns:
        bool: True if faces match, False otherwise.
    """
    results = face_recognition.compare_faces(
        [stored_encoding], live_encoding, tolerance=tolerance
    )
    return results[0]


def face_distance(live_encoding, stored_encoding):
    """
    Get the numeric distance between two face encodings.
    Useful for showing a confidence score (lower = more similar).

    Args:
        live_encoding (np.ndarray): Live face encoding.
        stored_encoding (np.ndarray): Stored face encoding.

    Returns:
        float: Distance value. Typically < 0.6 means same person.
    """
    distances = face_recognition.face_distance([stored_encoding], live_encoding)
    return float(distances[0])
