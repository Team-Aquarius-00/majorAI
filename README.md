# Hiring Project — Proctoring Modules

Extracted from **FuturProctor** — 6 standalone AI-powered monitoring modules ready to integrate into any project.

---

## 📁 Files

| File                      | Purpose                                                   |
| ------------------------- | --------------------------------------------------------- |
| `object_detection.py`     | Detect cell phones, books, multiple persons via YOLOv11   |
| `facial_detection.py`     | Count faces + draw landmarks via MediaPipe                |
| `gaze_tracking.py`        | Detect if user looks left/right/center via MediaPipe      |
| `audio_detection.py`      | Detect speaking/sound via PyAudio threshold               |
| `face_verification.py`    | Encode + compare faces for identity verification at login |
| `tab_switch_detection.js` | Frontend JS — detect/count/block tab switching            |

---

## ⚙️ Setup

### 1. Install Python 3.10 (required for dlib wheel)

### 2. Install dlib manually (must be done FIRST)

```bash
pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
```

(The .whl file is in the `AIExam/futurproctor/` folder)

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Copy the YOLO model

Copy `yolo11s.pt` from `AIExam/futurproctor/` into this folder.
`object_detection.py` expects it at the path you configure in `MODEL_PATH`.

---

## 🚀 Quick Usage

### Object Detection

```python
import cv2
from object_detection import detectObject

frame = cv2.imread("test.jpg")
labels, annotated_frame, person_count, detected_objects = detectObject(frame)
print(detected_objects)  # e.g. ["cell phone", "person"]
```

### Face Detection

```python
import cv2
from facial_detection import detectFace

frame = cv2.imread("test.jpg")
face_count, annotated_frame = detectFace(frame)
print(f"{face_count} face(s) detected")
```

### Gaze Tracking

```python
import cv2
from gaze_tracking import gaze_tracking

frame = cv2.imread("test.jpg")
result = gaze_tracking(frame)
print(result["gaze"])  # "left", "right", or "center"
```

### Audio Monitoring

```python
from audio_detection import audio_detection

result = audio_detection()  # Blocks until sound stops
if result["audio_detected"]:
    audio_bytes = result["audio_data"]  # Raw PCM bytes
```

### Face Verification

```python
import cv2
from face_verification import get_face_encoding, match_face_encodings
import numpy as np

# At registration — encode and store this
image = cv2.imread("user_photo.jpg")
encoding = get_face_encoding(image)         # Returns numpy array or None

# At login — compare live capture with stored encoding
live_image = cv2.imread("live_capture.jpg")
live_encoding = get_face_encoding(live_image)
stored_encoding = np.array(encoding)        # Load from your DB
is_match = match_face_encodings(live_encoding, stored_encoding)
print("Access granted" if is_match else "Face mismatch")
```

### Tab Switch Detection (Frontend)

Include `tab_switch_detection.js` in your HTML page and call:

```html
<script src="tab_switch_detection.js"></script>
<script>
  initTabSwitchDetection({
    maxSwitches: 5,
    onSwitch: count => console.log("Tab switch #" + count),
    onTerminate: () => alert("Too many tab switches! Session ended."),
    recordUrl: "/your-api/record-tab-switch/", // optional backend endpoint
    csrfToken: "YOUR_CSRF_TOKEN", // optional
  });
</script>
```
