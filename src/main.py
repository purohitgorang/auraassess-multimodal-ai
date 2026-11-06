import os
import sys
import time
import argparse
import threading
import cv2
import numpy as np

# Import submodules
from video.capture import VideoCaptureThread
from video.face_features import FaceFeatureExtractor
from video.gaze import GazeBlinkTracker
from video.pose_features import PoseFeatureExtractor
from audio.capture import AudioCaptureThread
from audio.acoustic import AcousticProsodyAnalyzer
from audio.latency import SpeechLatencyTracker
from fusion.calibration import CalibrationEngine
from fusion.aggregator import MultimodalAggregator
from model.predictor import BehaviorOutcomePredictor

class AuraAssessController:
    def __init__(self, video_src=0, audio_src=None, output_path="output/session_data.csv", calibrate_sec=15.0):
        self.video_src = video_src
        self.audio_src = audio_src
        self.output_path = output_path
        self.calibrate_sec = calibrate_sec

        # Initialize engines
        self.calibrator = CalibrationEngine(calibration_duration_sec=self.calibrate_sec)
        self.aggregator = MultimodalAggregator(self.calibrator, log_filepath=self.output_path)
        self.predictor = BehaviorOutcomePredictor()

        # Extractors
        self.face_extractor = FaceFeatureExtractor()
        self.gaze_tracker = GazeBlinkTracker()
        self.pose_extractor = PoseFeatureExtractor()
        self.audio_analyzer = AcousticProsodyAnalyzer()
        self.latency_tracker = SpeechLatencyTracker()

        # Threading workers
        self.cap_thread = None
        self.audio_thread = None
        self.running = False

    def _video_worker_loop(self):
        """Worker thread processing video frames as fast as possible (~30 FPS)"""
        print("[Video Worker] Starting facial and posture landmark pipeline...")
        while self.running:
            grabbed, frame = self.cap_thread.read()
            if not grabbed:
                time.sleep(0.01)
                continue

            h, w, _ = frame.shape

            # 1. Face landmarks
            face_features = self.face_extractor.extract_features(frame)
            
            # 2. Gaze and blinks
            gaze_features = None
            if face_features is not None:
                # Re-run face process internally or pass raw coordinates
                # Since face_extractor uses face mesh, let's extract landmarks
                # We can refine face_extractor to return raw landmarks list
                results = self.face_extractor.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0].landmark
                    gaze_features = self.gaze_tracker.process_face(face_landmarks, w, h)

            # 3. Pose features
            pose_features = self.pose_extractor.extract_features(frame)

            # 4. Merge frame features
            frame_features = {}
            if face_features:
                frame_features.update(face_features)
            if gaze_features:
                frame_features.update(gaze_features)
            if pose_features:
                frame_features.update(pose_features)

            # 5. Push to aggregator buffer
            if frame_features:
                self.aggregator.add_video_frame(frame_features)

            # Optional: Display output frame if running locally (not headlessly)
            # To run headlessly in server environments, we skip cv2.imshow
            if isinstance(self.video_src, int) and os.environ.get('DISPLAY') is not None:
                cv2.imshow("AuraAssess Feed (Press Q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

            time.sleep(0.02) # Cap at ~50 FPS to conserve CPU

        if os.environ.get('DISPLAY') is not None:
            cv2.destroyAllWindows()
        print("[Video Worker] Stopped.")

    def run(self):
        self.running = True

        # Start Video capture
        try:
            self.cap_thread = VideoCaptureThread(self.video_src)
            self.cap_thread.start()
        except Exception as e:
            print(f"Error starting Video Capture: {e}")
            return

        # Start Audio capture
        try:
            self.audio_thread = AudioCaptureThread(self.audio_src)
            self.audio_thread.start()
        except Exception as e:
            print(f"Error starting Audio Capture: {e}. Running in video-only mode.")

        # Start background video processor thread
        video_thread = threading.Thread(target=self._video_worker_loop)
        video_thread.start()

        # Start calibration phase
        start_time = time.time()
        self.calibrator.start(start_time)
        print(f"\n==========================================")
        print(f"  Starting baseline calibration ({self.calibrate_sec}s)...")
        print(f"  Please look directly at camera and speak calmly.")
        print(f"==========================================\n")

        # Core aggregation loop running once per second
        second_counter = 0
        try:
            while self.running:
                # Stop if video stream has finished (e.g. video file EOF)
                if self.cap_thread.stopped:
                    print("Video stream ended.")
                    break

                time.sleep(1.0)
                second_counter += 1
                current_time = time.time()

                # 1. Grab 1-second audio chunk and analyze prosody
                audio_chunk = np.zeros(0)
                if self.audio_thread:
                    audio_chunk = self.audio_thread.get_audio_chunk(duration_sec=1.0)
                
                audio_features = self.audio_analyzer.extract_features(audio_chunk)

                # 2. Track speech latency
                latency = None
                if len(audio_chunk) > 0:
                    # In real mock interview, trigger_prompt_end would be called by the system.
                    # For this standalone test, we simulate prompt ends periodically (every 10s)
                    if second_counter % 10 == 0:
                        print("\n[System] Simulating Question Prompt... Please answer!")
                        self.latency_tracker.trigger_prompt_end()
                    
                    latency = self.latency_tracker.process_audio_block(audio_chunk)

                # 3. Aggregate second features & calibrate
                row = self.aggregator.aggregate_second(current_time, audio_features, speech_latency=latency)

                # 4. Predict behavior state
                if not row.get("is_calibrating", True):
                    state, probs = self.predictor.predict(row)
                    print(f"[Sec {second_counter}] Behavior: \033[92m{state:<13}\033[0m | "
                          f"Stress: {probs['Stress']*100:3.0f}% | Chill: {probs['Chill']*100:3.0f}% | "
                          f"Confident: {probs['Confident']*100:3.0f}% | Overconfident: {probs['Overconfident']*100:3.0f}%")
                else:
                    print(f"[Sec {second_counter}] Calibrating... {second_counter}/{int(self.calibrate_sec)}s")

        except KeyboardInterrupt:
            print("\nInterrupt received. Stopping...")

        finally:
            self.running = False
            video_thread.join()
            
            self.cap_thread.stop()
            self.cap_thread.join()
            
            if self.audio_thread:
                self.audio_thread.stop()
            
            self.face_extractor.close()
            self.pose_extractor.close()

            # Save the session CSV logs
            self.aggregator.save_session()
            print("Cleanup complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AuraAssess Standalone Run")
    parser.add_argument("--video", type=str, default="0", help="Video source: index or path to file")
    parser.add_argument("--audio", type=str, default=None, help="Audio source: path to WAV file (optional)")
    parser.add_argument("--output", type=str, default="output/session_data.csv", help="Output path for session log")
    parser.add_argument("--calibrate", type=float, default=10.0, help="Calibration duration in seconds")

    args = parser.parse_args()

    # Parse video source (string digits mean camera index)
    v_src = args.video
    if v_src.isdigit():
        v_src = int(v_src)

    controller = AuraAssessController(
        video_src=v_src,
        audio_src=args.audio,
        output_path=args.output,
        calibrate_sec=args.calibrate
    )
    controller.run()
