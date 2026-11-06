import numpy as np
import time
import pandas as pd
import os

class MultimodalAggregator:
    def __init__(self, calibration_engine, log_filepath="session_features.csv"):
        self.calibrator = calibration_engine
        self.log_filepath = log_filepath
        self.session_data = []

        # Buffer arrays for sub-second video frame features
        self.video_buffer = {
            "furrow_index": [],
            "raise_index": [],
            "ear": [],
            "sway_score": [],
            "lateral_lean": [],
            "forward_lean": [],
            "pacifying_active": [],
            "hand_velocity": [],
            "gesture_score": [],
            "crossed_arms": [],
            "blink_rate_bpm": [],
            "gaze_deviation": []
        }

    def add_video_frame(self, video_features):
        """Adds frame-level video features to the aggregation buffer."""
        if video_features is None:
            return
        for key in self.video_buffer:
            if key in video_features and video_features[key] is not None:
                self.video_buffer[key].append(video_features[key])

    def aggregate_second(self, timestamp, audio_features=None, speech_latency=None):
        """
        Aggregates the accumulated video frames and audio features.
        Normalizes the results using the calibration engine.
        Returns:
            fused_record: dict (complete aggregated and normalized 1-second vector).
        """
        # 1. Average video features accumulated during the second
        aggregated_video = {}
        for key, values in self.video_buffer.items():
            if len(values) > 0:
                if key == "pacifying_active" or key == "crossed_arms":
                    # For boolean flags, if active at any point during this second, set to True
                    aggregated_video[key] = float(np.any(values))
                else:
                    aggregated_video[key] = float(np.mean(values))
            else:
                # Default fallback values
                aggregated_video[key] = 0.0 if key != "ear" else 0.25

        # Clear the video frame buffer for the next second
        for key in self.video_buffer:
            self.video_buffer[key].clear()

        # Extract audio features
        if audio_features is None:
            audio_features = {}
        pitch_mean = audio_features.get("pitch_mean", 0.0)
        pitch_std = audio_features.get("pitch_std", 0.0)
        energy_mean = audio_features.get("energy_mean", 0.0)
        energy_std = audio_features.get("energy_std", 0.0)
        latency_val = speech_latency if speech_latency is not None else 0.0

        # 2. Create raw baseline dictionary for calibration updating (video + audio)
        raw_calib = {
            "furrow_index": aggregated_video["furrow_index"],
            "raise_index": aggregated_video["raise_index"],
            "ear": aggregated_video["ear"],
            "sway_score": aggregated_video["sway_score"],
            "hand_velocity": aggregated_video["hand_velocity"],
            "gesture_score": aggregated_video["gesture_score"],
            "pitch_mean": pitch_mean,
            "energy_mean": energy_mean
        }

        # 3. Check if still in calibration, otherwise normalize
        is_calibrating = not self.calibrator.calibrated
        if is_calibrating:
            # Feed raw calibration data
            self.calibrator.update_calibration(timestamp, raw_calib)
            z_scores = {f"{k}_z": 0.0 for k in raw_calib}
        else:
            # Calibrated, compute Z-scores
            z_scores = self.calibrator.normalize(raw_calib)

        # 4. Build final synchronized row
        fused_row = {
            "timestamp": float(timestamp),
            "is_calibrating": bool(is_calibrating),
            # Video metrics
            "furrow_index": float(aggregated_video["furrow_index"]),
            "raise_index": float(aggregated_video["raise_index"]),
            "ear": float(aggregated_video["ear"]),
            "sway_score": float(aggregated_video["sway_score"]),
            "lateral_lean": float(aggregated_video["lateral_lean"]),
            "forward_lean": float(aggregated_video["forward_lean"]),
            "pacifying_active": float(aggregated_video["pacifying_active"]),
            "hand_velocity": float(aggregated_video["hand_velocity"]),
            "gesture_score": float(aggregated_video["gesture_score"]),
            "crossed_arms": float(aggregated_video["crossed_arms"]),
            "blink_rate_bpm": float(aggregated_video["blink_rate_bpm"]),
            "gaze_deviation": float(aggregated_video["gaze_deviation"]),
            # Audio metrics
            "pitch_mean": float(pitch_mean),
            "pitch_std": float(pitch_std),
            "energy_mean": float(energy_mean),
            "energy_std": float(energy_std),
            "speech_latency": float(latency_val),
            # Normalization Z-scores
            **z_scores,
            "crossed_arms_z": float(aggregated_video["crossed_arms"]) # Pass directly as Z-score equivalent
        }

        self.session_data.append(fused_row)
        return fused_row

    def save_session(self):
        """Saves accumulated second-by-second records to a CSV file."""
        if len(self.session_data) == 0:
            return
        df = pd.DataFrame(self.session_data)
        
        # Ensure target directory exists
        dir_name = os.path.dirname(self.log_filepath)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
        df.to_csv(self.log_filepath, index=False)
        print(f"Session features successfully written to: {self.log_filepath}")

if __name__ == "__main__":
    from calibration import CalibrationEngine
    print("Testing MultimodalAggregator...")
    calibrator = CalibrationEngine(calibration_duration_sec=2.0)
    aggregator = MultimodalAggregator(calibrator)
    
    # Start calibrator
    calibrator.start(0.0)
    
    # Feed mock video frame logs
    for _ in range(30):
        aggregator.add_video_frame({"ear": 0.26, "furrow_index": 0.5, "sway_score": 0.01})
        
    # Aggregate first second
    row = aggregator.aggregate_second(1.0, {"pitch_mean": 120.0, "energy_mean": 0.05})
    print(f"Aggregated Second 1: {row}")
