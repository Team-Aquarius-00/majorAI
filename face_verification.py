import cv2
from deepface import DeepFace


def capture_image(filename):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access camera")
        return None

    print("Press SPACE to capture image")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capture", frame)

        key = cv2.waitKey(1)

        if key == 32:  # SPACE key
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")
            break
        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    return filename


def verify_with_webcam(stored_image):
    # Capture live image
    live_image = capture_image("live_capture.jpg")

    if live_image is None:
        return

    try:
        result = DeepFace.verify(
            img1_path=stored_image,
            img2_path=live_image,
            model_name="ArcFace",
            enforce_detection=True,
        )

        print("\n===== RESULT =====")
        print("Verified:", result["verified"])
        print("Distance:", result["distance"])

        if result["verified"]:
            print("✅ Access Granted")
        else:
            print("❌ Face does not match")

    except Exception as e:
        print("Error:", e)


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    verify_with_webcam("user_photo.jpg")
