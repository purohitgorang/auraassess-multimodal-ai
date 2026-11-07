import os
import sys
import cv2
import numpy as np
import base64
import json
import time
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import uvicorn

# Ensure src and project root are in sys.path
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJ_DIR not in sys.path:
    sys.path.insert(0, _PROJ_DIR)

# Import our custom components
from video.face_features import FaceFeatureExtractor
from video.gaze import GazeBlinkTracker
from video.pose_features import PoseFeatureExtractor
from fusion.calibration import CalibrationEngine
from fusion.aggregator import MultimodalAggregator
from model.predictor import BehaviorOutcomePredictor

app = FastAPI(title="AuraAssess Web Dashboard")

# Initialize global modules
calibrator = CalibrationEngine(calibration_duration_sec=15.0)  # 15s for quick web testing
aggregator = MultimodalAggregator(calibrator, log_filepath="output/web_session_log.csv")
predictor = BehaviorOutcomePredictor()

face_extractor = FaceFeatureExtractor()
gaze_tracker = GazeBlinkTracker()
pose_extractor = PoseFeatureExtractor()

# Lock for multithreaded aggregator calls
fused_lock = threading.Lock()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket] Client connected.")
    
    # State tracking variables for client connection
    start_time = time.time()
    calibrator.start(start_time)
    
    last_agg_time = time.time()
    second_counter = 0

    try:
        while True:
            # Receive frame payload from browser client
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            msg_type = payload.get("type")
            
            if msg_type == "calibrate_reset":
                # Restart calibration
                start_time = time.time()
                calibrator.start(start_time)
                last_agg_time = time.time()
                second_counter = 0
                await websocket.send_json({"type": "status", "message": "Calibration restarted."})
                continue
                
            elif msg_type == "frame":
                img_data_b64 = payload.get("image")
                if not img_data_b64:
                    continue
                
                # Strip base64 header if present
                if "," in img_data_b64:
                    img_data_b64 = img_data_b64.split(",")[1]
                
                # Decode base64 frame to OpenCV Image
                img_bytes = base64.b64decode(img_data_b64)
                np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue

                h, w, _ = frame.shape

                # 1. Process Video features
                face_features = face_extractor.extract_features(frame)
                
                # Track face landmarks list to feed gaze tracker
                gaze_features = None
                raw_landmarks_serializable = []
                if face_features is not None:
                    results = face_extractor.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0].landmark
                        gaze_features = gaze_tracker.process_face(face_landmarks, w, h)
                        # Extract landmark subsets to send back to client for UI canvas rendering
                        # Send eye boundaries and silhouette points to keep it light
                        draw_indices = [33, 133, 159, 145, 263, 362, 386, 374, 70, 300, 468, 473]
                        for idx in draw_indices:
                            lm = face_landmarks[idx]
                            raw_landmarks_serializable.append({"x": lm.x, "y": lm.y, "idx": idx})

                pose_features = pose_extractor.extract_features(frame)
                
                # Add pose keypoints for client canvas rendering
                raw_pose_serializable = []
                if pose_features is not None:
                    results = pose_extractor.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.pose_landmarks:
                        pose_landmarks = results.pose_landmarks.landmark
                        # Draw nose, shoulders, wrists, hips
                        draw_pose_indices = [0, 11, 12, 15, 16, 23, 24]
                        for idx in draw_pose_indices:
                            lm = pose_landmarks[idx]
                            raw_pose_serializable.append({"x": lm.x, "y": lm.y, "idx": idx})

                # Compile sub-second video frame parameters
                frame_features = {}
                if face_features:
                    frame_features.update(face_features)
                if gaze_features:
                    frame_features.update(gaze_features)
                if pose_features:
                    frame_features.update(pose_features)

                # Push raw frame values to aggregator
                with fused_lock:
                    aggregator.add_video_frame(frame_features)

                # 2. Check if a full second has elapsed to run prediction
                now = time.time()
                if now - last_agg_time >= 1.0:
                    second_counter += 1
                    last_agg_time = now

                    # Aggregate second and normalize Z-scores
                    with fused_lock:
                        fused_row = aggregator.aggregate_second(now)

                    # Calculate state predictions
                    state = "Calibrating"
                    probs = {"Stress": 0.25, "Chill": 0.25, "Confident": 0.25, "Overconfident": 0.25}
                    
                    if not fused_row.get("is_calibrating", True):
                        state, probs = predictor.predict(fused_row)

                    # Send unified second update back to browser
                    await websocket.send_json({
                        "type": "second_update",
                        "second": second_counter,
                        "state": state,
                        "probabilities": probs,
                        "features": fused_row
                    })

                # Send frame-level response immediately (used for rendering tracking meshes in real-time)
                await websocket.send_json({
                    "type": "frame_response",
                    "landmarks": raw_landmarks_serializable,
                    "pose_landmarks": raw_pose_serializable,
                    "pacifying_active": frame_features.get("pacifying_active", False)
                })

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")

# Mount Static Files folder for Web GUI
_WEB_DIR = os.path.join(_PROJ_DIR, "web")
app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")

@app.on_event("shutdown")
def shutdown_event():
    print("Stopping Cleanup Server...")
    face_extractor.close()
    pose_extractor.close()
    aggregator.save_session()

if __name__ == "__main__":
    # Run FastAPI web app using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
