import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

from src.fusion.generator import SyntheticInterviewGenerator
from src.model.predictor import BehaviorOutcomePredictor

def main():
    dataset_path = "synthetic_interview_data.csv"
    model_path = "behavior_classifier.pkl"

    # 1. Generate synthetic dataset
    print("Step 1: Generating high-fidelity synthetic interview dataset with advanced kinesics...")
    generator = SyntheticInterviewGenerator()
    generator.generate_dataset(output_csv=dataset_path, num_sessions_per_state=50)

    # Load data
    df = pd.read_csv(dataset_path)

    # 2. Extract features and target labels (unified multimodal parameters)
    feature_keys = [
        "furrow_index", "raise_index", "ear", "sway_score", 
        "lateral_lean", "forward_lean", "pacifying_active",
        "gaze_deviation", "hand_velocity", "gesture_score", "crossed_arms",
        "pitch_mean", "pitch_std", "energy_mean", "energy_std", "speech_latency",
        "furrow_index_z", "raise_index_z", "ear_z", "sway_score_z",
        "hand_velocity_z", "gesture_score_z", "crossed_arms_z",
        "pitch_mean_z", "energy_mean_z"
    ]

    # Map text labels to indices
    label_map = {"Stress": 0, "Chill": 1, "Confident": 2, "Overconfident": 3}
    inverse_label_map = {0: "Stress", 1: "Chill", 2: "Confident", 3: "Overconfident"}
    
    # 3. Perform a Group-based Train/Test Split (by session_id)
    # This prevents frames from the same interview session from bleeding into both sets.
    print("\nStep 2: Preparing session-level Train/Test split...")
    unique_sessions = df["session_id"].unique()
    train_sessions, test_sessions = train_test_split(unique_sessions, test_size=0.20, random_state=42)

    train_df = df[df["session_id"].isin(train_sessions)]
    test_df = df[df["session_id"].isin(test_sessions)]

    X_train = train_df[feature_keys].values
    y_train = np.array([label_map[l] for l in train_df["label"].values])
    
    X_test = test_df[feature_keys].values
    y_test = np.array([label_map[l] for l in test_df["label"].values])

    print(f"  Training samples: {len(X_train)} rows ({len(train_sessions)} sessions)")
    print(f"  Testing samples:  {len(X_test)} rows ({len(test_sessions)} sessions)")

    # 4. Train the Predictor
    print("\nStep 3: Training Behavior Predictor (Random Forest)...")
    predictor = BehaviorOutcomePredictor(model_path=model_path)
    
    # We train the classifier inside the predictor wrapper
    predictor.clf.fit(X_train, y_train)
    predictor.is_trained = True
    
    # Save the trained model weights
    joblib.dump(predictor.clf, model_path)
    print(f"  Model saved to: {model_path}")

    # 5. Evaluate on Unseen Test Sessions
    print("\nStep 4: Running model evaluation on test sessions...")
    y_pred = predictor.clf.predict(X_test)
    y_pred_proba = predictor.clf.predict_proba(X_test)

    # Compute Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n==========================================")
    print(f"  CLASSIFICATION ACCURACY: {accuracy * 100:.2f}%")
    print(f"==========================================\n")

    # Print full classification report
    target_names = ["Stress", "Chill", "Confident", "Overconfident"]
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Print Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df)
    
    print("\nTraining workflow complete! Real-time predictions will now use this ML classifier.")

if __name__ == "__main__":
    main()
