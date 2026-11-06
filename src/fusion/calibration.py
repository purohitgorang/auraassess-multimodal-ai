import numpy as np

class CalibrationEngine:
    def __init__(self, calibration_duration_sec=60.0):
        self.calibration_duration = calibration_duration_sec
        self.start_time = None
        self.calibrated = False

        # Accumulation lists for baseline features
        self.history = {
            "furrow_index": [],
            "raise_index": [],
            "ear": [],
            "sway_score": [],
            "hand_velocity": [],
            "gesture_score": [],
            "pitch_mean": [],
            "energy_mean": []
        }

        # Stored statistical parameters (means & standard deviations)
        self.baselines = {}

    def start(self, current_time):
        self.start_time = current_time
        self.calibrated = False
        for key in self.history:
            self.history[key].clear()

    def update_calibration(self, current_time, raw_features):
        """
        Accumulates raw features during the calibration phase.
        Returns:
            calibrated: bool (True if calibration just finished, False otherwise)
        """
        if self.calibrated or self.start_time is None:
            return False

        # Append feature values if present
        for key, val in raw_features.items():
            if key in self.history and val is not None:
                self.history[key].append(val)

        # Check if duration elapsed
        elapsed = current_time - self.start_time
        if elapsed >= self.calibration_duration:
            self._compute_baselines()
            self.calibrated = True
            return True

        return False

    def _compute_baselines(self):
        """Computes means and standard deviations for all collected features."""
        for key, values in self.history.items():
            if len(values) > 5:
                mean = np.mean(values)
                std = np.std(values)
                if std == 0:
                    std = 1.0  # Avoid division by zero
                self.baselines[key] = {"mean": float(mean), "std": float(std)}
            else:
                # Fallbacks in case of sparse data
                self.baselines[key] = {"mean": 0.0, "std": 1.0}
        print("Calibration complete. Baseline statistics established:")
        for key, stats in self.baselines.items():
            print(f"  {key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    def normalize(self, raw_features):
        """
        Transforms raw feature inputs into Z-score normalized outputs relative to baselines.
        Formula: Z = (X - Mean) / StdDev
        """
        if not self.calibrated:
            # Return raw values or neutral Z-scores if not yet calibrated
            return {f"{k}_z": 0.0 for k in self.history}

        normalized = {}
        for key, val in raw_features.items():
            if key in self.baselines:
                mean = self.baselines[key]["mean"]
                std = self.baselines[key]["std"]
                if val is not None:
                    z_score = (val - mean) / std
                else:
                    z_score = 0.0  # Neutral default for missing frames
                normalized[f"{key}_z"] = float(z_score)
        
        return normalized

if __name__ == "__main__":
    print("Testing CalibrationEngine...")
    engine = CalibrationEngine(calibration_duration_sec=3.0)
    
    # Simulate 3 seconds of data
    start = 100.0
    engine.start(start)
    
    # Feed mock calm data (e.g. EAR around 0.25, sway around 0.01)
    for t in np.linspace(100.0, 103.5, 35):
        raw = {
            "furrow_index": 0.5 + 0.02 * np.random.randn(),
            "raise_index": 0.1 + 0.01 * np.random.randn(),
            "ear": 0.25 + 0.01 * np.random.randn(),
            "sway_score": 0.01 + 0.002 * np.random.randn(),
            "pitch_mean": 120.0 + 2.0 * np.random.randn(),
            "energy_mean": 0.05 + 0.005 * np.random.randn()
        }
        engine.update_calibration(t, raw)

    print(f"Is calibrated? {engine.calibrated}")
    
    # Try normalization on a stressed state (furrow indices drops, sway increases, pitch spikes)
    stress_raw = {
        "furrow_index": 0.38, # much lower (eyebrows furrowed)
        "raise_index": 0.1,
        "ear": 0.24,
        "sway_score": 0.05,  # much higher (rocking)
        "pitch_mean": 140.0, # voice pitch spike
        "energy_mean": 0.05
    }
    
    z_scores = engine.normalize(stress_raw)
    print("Normalized Stressed Features:")
    for k, v in z_scores.items():
        print(f"  {k}: {v:+.2f}")
