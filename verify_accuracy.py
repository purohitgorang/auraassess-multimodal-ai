import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def test_saved_model_accuracy():
    model_path = "behavior_classifier.pkl"
    dataset_path = "synthetic_interview_data.csv"

    print("==================================================")
    print("        AURAASSESS ACCURACY DIAGNOSTIC RUN        ")
    print("==================================================")

    # 1. Verify files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Run train_model.py first.")
        return
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' not found. Run train_model.py first.")
        return

    # 2. Load model and dataset
    clf = joblib.load(model_path)
    df = pd.read_csv(dataset_path)

    feature_keys = [
        "furrow_index", "raise_index", "ear", "sway_score", 
        "lateral_lean", "forward_lean", "pacifying_active",
        "gaze_deviation", "hand_velocity", "gesture_score", "crossed_arms",
        "pitch_mean", "pitch_std", "energy_mean", "energy_std", "speech_latency",
        "furrow_index_z", "raise_index_z", "ear_z", "sway_score_z",
        "hand_velocity_z", "gesture_score_z", "crossed_arms_z",
        "pitch_mean_z", "energy_mean_z"
    ]

    label_map = {"Stress": 0, "Chill": 1, "Confident": 2, "Overconfident": 3}
    target_names = ["Stress", "Chill", "Confident", "Overconfident"]

    # 3. Perform identical session-level split to evaluate test partition
    unique_sessions = df["session_id"].unique()
    _, test_sessions = train_test_split(unique_sessions, test_size=0.20, random_state=42)
    test_df = df[df["session_id"].isin(test_sessions)]

    X_test = test_df[feature_keys].values
    y_test = np.array([label_map[l] for l in test_df["label"].values])

    # 4. Predict
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 5. Output Report
    print(f"\nModel File Status: ACTIVE")
    print(f"Total Test Set Samples Evaluated: {len(X_test)} frames")
    print(f"Model Classification Accuracy: {accuracy * 100:.2f}%\n")
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("==================================================")

if __name__ == "__main__":
    test_saved_model_accuracy()
