import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class BehaviorOutcomePredictor:
    def __init__(self, model_path="behavior_classifier.pkl"):
        self.model_path = model_path
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

        # Load pre-trained model if exists
        if os.path.exists(model_path):
            try:
                self.clf = joblib.load(model_path)
                self.is_trained = True
                print(f"Loaded existing model from: {model_path}")
            except Exception as e:
                print(f"Failed to load model file, defaulting to heuristic. Error: {e}")

    def predict_heuristic(self, fused_row):
        """
        Applies a rule-based psychological heuristic to calculate probability scores
        for the four states: 'Stress', 'Chill', 'Confident', 'Overconfident'.
        """
        # Extract features
        furrow_z = fused_row.get("furrow_index_z", 0.0)
        raise_z = fused_row.get("raise_index_z", 0.0)
        sway_z = fused_row.get("sway_score_z", 0.0)
        ear = fused_row.get("ear", 0.25)
        pacifying = fused_row.get("pacifying_active", 0.0)
        gaze_dev = fused_row.get("gaze_deviation", 0.1)
        
        # New kinesic features
        hand_vel_z = fused_row.get("hand_velocity_z", 0.0)
        gesture_z = fused_row.get("gesture_score_z", 0.0)
        crossed_arms = fused_row.get("crossed_arms", 0.0)

        # Initialize scores
        scores = {
            "Stress": 0.0,
            "Chill": 0.0,
            "Confident": 0.0,
            "Overconfident": 0.0
        }

        # --- Stress Logic ---
        if furrow_z < -1.0:
            scores["Stress"] += 2.0 * abs(furrow_z)
        if pacifying > 0.5:
            scores["Stress"] += 2.5
        if sway_z > 1.2:
            scores["Stress"] += 2.0 * sway_z
        if ear < 0.18:
            scores["Stress"] += 1.5
        # Fidgeting wrists velocity
        if hand_vel_z > 1.2:
            scores["Stress"] += 1.5 * hand_vel_z
        # Tucked in arms (low gesturing)
        if gesture_z < -0.8:
            scores["Stress"] += 1.0
        # Defensive crossed arms
        if crossed_arms > 0.5:
            scores["Stress"] += 1.5

        # --- Chill Logic ---
        if sway_z < 0.5:
            scores["Chill"] += 1.5
        if pacifying < 0.5:
            scores["Chill"] += 1.5
        if -1.0 <= furrow_z <= 1.0:
            scores["Chill"] += 1.0
        # Calm hands velocity
        if hand_vel_z < 0.0:
            scores["Chill"] += 1.0
        # Natural gesture space
        if -0.5 <= gesture_z <= 0.5:
            scores["Chill"] += 1.0

        # --- Confident Logic ---
        if gaze_dev < 0.12:
            scores["Confident"] += 2.5
        if sway_z < 0.8:
            scores["Confident"] += 1.5
        if furrow_z > -0.5:
            scores["Confident"] += 1.0
        # Stable, non-fidgeting hand velocity
        if hand_vel_z < 0.8:
            scores["Confident"] += 1.0
        # Open, expressive hand gestures (not too wide, but active)
        if 0.5 < gesture_z < 1.8:
            scores["Confident"] += 2.0

        # --- Overconfident Logic ---
        forward_lean = fused_row.get("forward_lean", 0.0)
        if forward_lean < -0.3:
            scores["Overconfident"] += 2.0
        if raise_z > 1.2:
            scores["Overconfident"] += 1.5
        # Expansive, wide, space-taking hand gestures
        if gesture_z >= 1.8:
            scores["Overconfident"] += 2.0

        # Normalize scores
        exp_scores = {k: np.exp(v) for k, v in scores.items()}
        sum_exp = sum(exp_scores.values())
        probabilities = {k: float(v / sum_exp) for k, v in exp_scores.items()}
        predicted_state = max(probabilities, key=probabilities.get)

        return predicted_state, probabilities

    def predict(self, fused_row):
        """
        Predicts state. Uses machine learning model if trained, otherwise falls back to heuristics.
        """
        if not self.is_trained:
            return self.predict_heuristic(fused_row)

        # Unified 25-feature list (video + audio)
        feature_keys = [
            "furrow_index", "raise_index", "ear", "sway_score", 
            "lateral_lean", "forward_lean", "pacifying_active",
            "gaze_deviation", "hand_velocity", "gesture_score", "crossed_arms",
            "pitch_mean", "pitch_std", "energy_mean", "energy_std", "speech_latency",
            "furrow_index_z", "raise_index_z", "ear_z", "sway_score_z",
            "hand_velocity_z", "gesture_score_z", "crossed_arms_z",
            "pitch_mean_z", "energy_mean_z"
        ]
        
        # Default missing keys to 0.0 to handle visual-only (Web UI) vs multimodal gracefully
        x = [fused_row.get(k, 0.0) if fused_row.get(k) is not None else 0.0 for k in feature_keys]
        x = np.array(x).reshape(1, -1)

        pred_idx = self.clf.predict(x)[0]
        prob = self.clf.predict_proba(x)[0]

        states = ["Stress", "Chill", "Confident", "Overconfident"]
        predicted_state = states[pred_idx]
        probabilities = {states[i]: float(prob[i]) for i in range(len(states))}

        return predicted_state, probabilities

    def train_model(self, csv_filepath, label_column="label"):
        """
        Trains the classifier on labeled historical session logs.
        """
        if not os.path.exists(csv_filepath):
            print(f"Error: training file {csv_filepath} not found.")
            return False

        df = pd.read_csv(csv_filepath)
        
        # Unified 25-feature list (video + audio)
        feature_keys = [
            "furrow_index", "raise_index", "ear", "sway_score", 
            "lateral_lean", "forward_lean", "pacifying_active",
            "gaze_deviation", "hand_velocity", "gesture_score", "crossed_arms",
            "pitch_mean", "pitch_std", "energy_mean", "energy_std", "speech_latency",
            "furrow_index_z", "raise_index_z", "ear_z", "sway_score_z",
            "hand_velocity_z", "gesture_score_z", "crossed_arms_z",
            "pitch_mean_z", "energy_mean_z"
        ]

        # Extract features and targets
        # Fill NaN values with 0.0 if any are missing
        df_filled = df.fillna(0.0)
        X = df_filled[feature_keys].values
        y_raw = df[label_column].values

        # Map labels to indices
        label_map = {"Stress": 0, "Chill": 1, "Confident": 2, "Overconfident": 3}
        y = np.array([label_map.get(lbl, 1) for lbl in y_raw])

        # Train model
        self.clf.fit(X, y)
        self.is_trained = True
        
        # Save model weights
        joblib.dump(self.clf, self.model_path)
        print(f"Model successfully trained and saved to: {self.model_path}")
        return True

if __name__ == "__main__":
    print("Testing BehaviorOutcomePredictor Heuristics...")
    predictor = BehaviorOutcomePredictor()
    
    # Stress input vector (furrow index low, sway index high, self-touch active)
    stress_vector = {
        "furrow_index_z": -2.1,
        "raise_index_z": 0.0,
        "sway_score_z": 1.8,
        "ear": 0.17,
        "pacifying_active": 1.0,
        "gaze_deviation": 0.15,
        "pitch_std": 14.5,
        "speech_latency": 4.2,
        "forward_lean": -0.1
    }
    
    state, probs = predictor.predict(stress_vector)
    print(f"Predicted State: {state}")
    print("State Probabilities:")
    for k, v in probs.items():
        print(f"  {k}: {v*100:.1f}%")
