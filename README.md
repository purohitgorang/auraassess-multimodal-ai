# 🧠 AuraAssess: Experimental Multimodal Telemetry & Self-Rehearsal Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/FastAPI-0.110%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Computer Vision](https://img.shields.io/badge/MediaPipe-FaceMesh%20%26%20Pose-orange.svg)](https://mediapipe.dev/)

**AuraAssess** is an experimental multimodal machine learning system that fuses real-time facial landmark geometry, upper-body posture angles, and acoustic prosody into streaming telemetry. Built as an **interactive self-rehearsal tool**, it helps speakers and candidates practice presentation pacing, posture stability, and vocal modulation in real time.

---

> [!IMPORTANT]
> ### 🛡️ Ethical AI, Bias Governance & Scientific Limitations
> **AuraAssess is strictly an experimental self-coaching and research tool. It is explicitly NOT designed, validated, or legally compliant for automated hiring gates, candidate ranking, or personality assessment.**
>
> 1. **Scientific Ground Truth:** Micro-expressions and body telemetry do not equate to job competence, honesty, or capability. Internal emotional states have significant variance and cannot be definitively inferred from visual landmark coordinates alone.
> 2. **Neurodivergence & Accessibility:** Candidates with neurodivergent conditions (e.g., autism, ADHD, Tourette syndrome, motor tics) or physical facial differences (e.g., Bell's palsy) naturally exhibit unique eye contact and movement patterns that algorithms must never penalize.
> 3. **Cultural Context:** Communication dynamics—such as eye contact duration, smiling frequency, and vocal pitch—differ widely across global cultures. A standardized baseline cannot represent human diversity.
> 4. **Privacy-Preserving Design:** **Zero raw audio or video is stored on disk.** Frames and audio buffers are processed ephemerally in RAM and immediately discarded. Only rolling mathematical statistics are streamed over WebSockets.

---

## 📊 Telemetry Pipeline & Multimodal Feature Extraction

AuraAssess processes live sensory streams at **30 FPS** across three concurrent modalities:

| Modality | Extraction Engine | Extracted Features | Telemetry Payload |
|---|---|---|---|
| **Facial Kinematics** | MediaPipe FaceMesh | 468 3D facial landmarks | Eye Aspect Ratio (EAR), blink cadence, smile delta, eyebrow furrow variance |
| **Skeletal Posture** | MediaPipe BlazePose | 33 upper-body landmarks | Shoulder tilt angle, spinal slouch index, hand-to-face gesture frequency |
| **Acoustic Prosody** | PyAudio / Librosa DSP | Short-time Fourier transform | Root Mean Square (RMS) energy, pitch jitter, speaking rate, silence intervals |

### Baseline Normalization (First 15 Seconds)
To adjust for individual camera placement, ambient lighting, and individual resting facial structure, the engine computes a running mean ($\mu$) and standard deviation ($\sigma$) during an initial 15-second calibration window, scoring subsequent telemetry against individualized Z-scores rather than arbitrary global thresholds.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│                   Optical & Audio Ingestion            │
│         Webcam Stream (30 FPS) + Microphone (16kHz)     │
└───────────────────────────┬────────────────────────────┘
                            │ Raw Buffers (In-Memory Only)
                            ▼
┌────────────────────────────────────────────────────────┐
│             Multimodal Feature Extraction              │
│   • 468-point FaceMesh (Geometric Landmark Vectors)   │
│   • 33-point BlazePose (Shoulder/Spine Angle Vectors) │
│   • Acoustic Prosody (RMS Energy & Zero-Crossing)      │
└───────────────────────────┬────────────────────────────┘
                            │ Normalized Feature Vector
                            ▼
┌────────────────────────────────────────────────────────┐
│           Individualized Z-Score Normalizer            │
│         Running Mean & Variance Calibration Buffer     │
└───────────────────────────┬────────────────────────────┘
                            │ Standardized Telemetry Frame
                            ▼
┌────────────────────────────────────────────────────────┐
│             Full-Duplex WebSocket Engine (Port 8080)   │
│         FastAPI Server + Interactive Rehearsal UI      │
└────────────────────────────────────────────────────────┘
```

---

## 💡 Engineering Ownership & Design Decisions

### 1. What problem was identified?
Public speaking and interview anxiety affect millions of candidates. Traditional human coaching is expensive and inaccessible, while cloud-based AI tools require uploading sensitive personal video recordings to third-party servers. AuraAssess was created to provide free, real-time, privacy-first local telemetry for rehearsal.

### 2. Why this architecture?
* **Edge / Local Processing:** Processing MediaPipe models directly on the client machine ensures zero personal video data ever touches a remote server.
* **Full-Duplex WebSockets over REST:** Polling REST endpoints creates severe HTTP overhead and visual stutter. A persistent WebSocket connection streams JSON telemetry packets with sub-10ms latency at 30 FPS.

### 3. What failed during development & how was it fixed?
* **Failure:** Audio and video streams experienced progressive synchronization drift over prolonged sessions due to differing OS audio buffer sizes and camera frame drops.
* **Solution:** Decoupled audio DSP into an independent circular ring buffer with monotonically increasing monotonic timestamps, synchronizing audio energy windows with the nearest camera frame timestamp.

### 4. What would be added in a production version?
* WebAssembly (Wasm) client-side compilation to eliminate Python backend requirements entirely, running 100% inside a standard browser tab.
* Longitudinal progress tracking allowing candidates to visualize their posture stability and vocal pacing improvements across multiple practice sessions.

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/purohitgorang/auraassess-multimodal-ai.git
cd auraassess-multimodal-ai

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Rehearsal Server
```bash
python main.py
```
Open your browser to: **`http://localhost:8080`**

---

## 📄 License
MIT License. Copyright (c) 2026 Gorang Purohit.
