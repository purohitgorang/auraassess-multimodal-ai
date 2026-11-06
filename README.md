# 🧠 AuraAssess: Multimodal Behavioral & Emotion Intelligence Engine

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/FastAPI-0.110%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Computer Vision](https://img.shields.io/badge/MediaPipe-FaceMesh%20%26%20Pose-orange.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

AuraAssess is an end-to-end multimodal machine learning system designed to evaluate real-time candidate composure, behavioral states, and communication dynamics during technical and executive interviews.

---

## 🌟 Key Features

* **Multimodal Data Fusion:** Extracts features simultaneously from facial geometry (MediaPipe Face Mesh), upper-body pose & posture shifts (BlazePose), and audio speech prosody.
* **4-State Behavioral Classification:**
  * **Confident:** Stable gaze, erect posture, smooth speech cadence.
  * **Chill / Relaxed:** Low muscle tension, steady blink rates, comfortable pacing.
  * **Stressed / Nervous:** Elevated blink rate, pacifying hand gestures, vocal pitch shifts.
  * **Overconfident / Defensive:** Rigid posture, sustained aggressive gaze, atypical micro-expressions.
* **15-Second Baseline Calibration:** Automatically computes individualized Z-scores during the first 15 seconds to ensure cultural and individual fairness.
* **Bi-directional WebSocket Telemetry:** Low-latency 30 FPS stream delivering landmark tracking coordinates, rolling probabilities, and second-by-second analytics to an interactive browser GUI.

---

## 🏗️ Architecture

```
Webcam / Microphone ──> MediaPipe Video Pipeline  ──┐
                    ──> PyAudio / Librosa DSP     ──┼──> Multimodal Aggregator ──> ML Classifier ──> WebSocket API (Port 8080)
```

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/auraassess.git
cd auraassess

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Web Dashboard & Server
```bash
# Windows turnkey launcher
start_server.bat

# Or run manually via Python
python src/server.py
```
Open your browser to: **`http://localhost:8080`**

### 3. Retrain the Behavior Model
```bash
python train_model.py
python verify_accuracy.py
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📄 License
This project is licensed under the MIT License.
