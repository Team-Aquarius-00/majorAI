# 🎓 Hiring Proctoring System — AI-Powered Monitoring Modules

A comprehensive interview monitoring and proctoring system with 5 standalone AI modules for detecting cheating, unauthorized devices, and suspicious behavior in real-time.

---

## 📋 Modules Overview

| Module                   | File                      | Purpose                                               |
| ------------------------ | ------------------------- | ----------------------------------------------------- |
| **Object Detection**     | `object_detection.py`     | Detect cell phones, books, multiple persons (YOLOv11) |
| **Facial Detection**     | `facial_detection.py`     | Detect & count faces, draw landmarks (MediaPipe)      |
| **Gaze Tracking**        | `newgaze.py`              | Monitor eye gaze direction & head pose (MediaPipe)    |
| **Audio Recording**      | `audio_detection.py`      | Record system + microphone audio, detect speaking     |
| **Face Verification**    | `face_verification.py`    | Facial recognition & identity verification (DeepFace) |
| **Tab Switch Detection** | `tab_switch_detection.js` | Browser-based tab switching detection (Frontend)      |

---

## 🔧 System Requirements

- **Python**: 3.10+ (required for dlib compatibility)
- **OS**: Windows 10+, macOS, or Linux
- **GPU**: (Optional) NVIDIA GPU with CUDA for faster inference
- **Microphone**: Required for audio recording module
- **Webcam**: Required for facial detection, gaze tracking, and verification

---

## 📦 Installation

### Step 1: Install Python 3.10

Download and install Python 3.10+ from [python.org](https://www.python.org)

Verify installation:

```bash
python --version
# Output should be: Python 3.10.x or 3.11.x, etc.
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
# Activate
venv\Scripts\activate         # Windows
source venv/bin/activate      # macOS/Linux
```

### Step 3: Install dlib Wheel (Windows + Python 3.10)

**Important**: Install dlib FIRST before running `requirements.txt`

```bash
# Windows + Python 3.10
pip install dlib-19.22.99-cp310-cp310-win_amd64.whl

# macOS/Linux (uses compiled version)
pip install dlib
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**

- `opencv-python` >= 4.11.0 — Video/image processing
- `mediapipe` >= 0.10.20 — Face & gaze detection
- `ultralytics` >= 8.3.62 — YOLOv11 object detection
- `face-recognition` >= 1.3.0 — Facial encoding & matching
- `deepface` >= 0.0.75 — Face verification
- `pyaudio` >= 0.2.14 — Audio recording
- `torch` >= 2.5.1 & `torchvision` >= 0.20.1 — ML backend

### Step 5: Copy YOLO Model

Copy the pre-trained YOLO model to the hiring folder:

```bash
# Source location (from AIExam/futurproctor/)
cp yolo11s.pt .
```

Verify:

```bash
ls yolo11s.pt  # Should show: yolo11s.pt
```

---

## 🚀 Running Each Module

### 1. Object Detection (Detect Phones, Books, People)

```bash
python object_detection.py
```

**Usage in code:**

```python
import cv2
from object_detection import detectObject

frame = cv2.imread("image.jpg")
labels, annotated_frame, person_count, detected_objects = detectObject(frame)
print(detected_objects)  # ["cell phone", "person", "book"]
```

---

### 2. Facial Detection (Detect Faces & Landmarks)

```bash
python facial_detection.py
```

**Usage in code:**

```python
import cv2
from facial_detection import detectFace

frame = cv2.imread("image.jpg")
face_count, annotated_frame = detectFace(frame)
print(f"Faces detected: {face_count}")
```

---

### 3. Gaze Tracking (Monitor Eye Gaze & Head Pose)

```bash
python newgaze.py
```

**Command-line options:**

```bash
python newgaze.py --source 0 --name "Candidate1" --out ./reports
python newgaze.py --headless  # Run without display
```

**Keyboard Controls:**

- `Q` — Quit and save report
- `R` — Reset session
- `S` — Save report without quitting
- `C` — Recalibrate gaze baseline

---

### 4. Audio Recording (Record System + Microphone Audio)

```bash
python audio_detection.py
```

**Features:**

- Records from both microphone AND system audio (speakers/headphone)
- Auto-saves to `call_recording.wav`
- Press `q` or `Ctrl+C` to stop

**Enable System Audio (Windows):**

1. Right-click Speaker icon → Sound settings
2. Go to "Volume mixer" or "App volume and device preferences"
3. Enable "Stereo Mix" (if not visible, enable in Device Manager)

---

### 5. Face Verification (Facial Recognition Login)

```bash
python face_verification.py
```

**Workflow:**

1. **Capture reference image** — Press SPACE to capture
2. **Verify with webcam** — Live face comparison
3. **Get match results** — Similar/Mismatch

---

### 6. Tab Switch Detection (Browser Monitoring)

Include in your HTML page:

```html
<script src="tab_switch_detection.js"></script>
<script>
  initTabSwitchDetection({
    maxSwitches: 5,
    onSwitch: count => console.log("Tab switch #" + count),
    onTerminate: () => alert("Too many tab switches! Session ended."),
  });
</script>
```

---

## 📊 Output Files

- **Audio Recording**: `call_recording.wav` (in hiring folder)
- **Gaze Tracking Report**: `reports/` folder (JSON & CSV)
- **Face Capture**: `captured_face.jpg`
- **Live Capture**: `live_capture.jpg`

---

## ⚠️ Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'mediapipe'"

**Solution**: Ensure all dependencies are installed:

```bash
pip install -r requirements.txt -U
```

### Issue: "FileNotFoundError: yolo11s.pt"

**Solution**: Copy the YOLO model file to the hiring folder:

```bash
cp ../AIExam/futurproctor/yolo11s.pt .
```

### Issue: "Failed to open camera" / "ModuleNotFoundError: No module named 'cv2'"

**Solution**:

```bash
pip install opencv-python
```

### Issue: Audio recording not working on Windows

**Solution**: Enable Stereo Mix:

1. Right-click Speaker → Sound settings
2. Advanced options → Volume mixer
3. Find and enable "Stereo Mix"

### Issue: dlib installation fails

**Solution**:

```bash
# Windows + Python 3.10
pip install dlib-19.22.99-cp310-cp310-win_amd64.whl

# If wheel not found, use precompiled:
pip install dlib --only-binary :all: -U
```

---

## 🔍 Integration Example

```python
import cv2
import threading
from object_detection import detectObject
from facial_detection import detectFace
from audio_detection import record_call_audio

# Start audio recording in background
audio_thread = threading.Thread(
    target=record_call_audio,
    kwargs={"output_file": "interview_record.wav"},
    daemon=True
)
audio_thread.start()

# Process video frames
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detections
    obj_labels, obj_frame, ppl_count, objects = detectObject(frame)
    face_count, face_frame = detectFace(frame)

    # Check for cheating signals
    if "cell phone" in objects or ppl_count > 1:
        print("⚠️ ALERT: Suspicious activity detected!")

    cv2.imshow("Monitoring", face_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📝 Configuration Files

Create `config.json` (optional) to override defaults:

```json
{
  "confidence_threshold": 0.5,
  "gaze_sensitivity": 0.7,
  "audio_threshold": 2000,
  "max_tab_switches": 5,
  "report_format": "json"
}
```

---

## 📄 License

Part of the FuturProctor project. All rights reserved © 2025.

---

## 🤝 Support

For issues or questions, check the module docstrings or contact the development team.
