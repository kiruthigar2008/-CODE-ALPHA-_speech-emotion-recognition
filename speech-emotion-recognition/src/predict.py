"""
predict.py
----------
Run inference on a single .wav file using the trained model.

Usage:
    python src/predict.py --file path/to/audio.wav
"""

import os
import argparse
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences

from feature_extraction import extract_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_artifacts():
    model = load_model(os.path.join(MODELS_DIR, "emotion_model.h5"))

    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
        artifacts = pickle.load(f)

    mean = np.load(os.path.join(MODELS_DIR, "norm_mean.npy"))
    std = np.load(os.path.join(MODELS_DIR, "norm_std.npy"))

    return model, artifacts["label_encoder"], artifacts["max_len"], mean, std


def predict_emotion(file_path: str):
    model, label_encoder, max_len, mean, std = load_artifacts()

    features = extract_features(file_path)
    features_padded = pad_sequences(
        [features], maxlen=max_len, dtype="float32", padding="post", truncating="post"
    )
    features_norm = (features_padded - mean) / std

    probs = model.predict(features_norm, verbose=0)[0]
    predicted_idx = np.argmax(probs)
    predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
    confidence = probs[predicted_idx] * 100

    print(f"\n🎧 Predicted Emotion: {predicted_label} (confidence: {confidence:.1f}%)\n")
    print("All class probabilities:")
    for label, p in sorted(zip(label_encoder.classes_, probs), key=lambda x: -x[1]):
        print(f"  {label:12s}: {p*100:5.1f}%")

    return predicted_label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict emotion from a speech audio file.")
    parser.add_argument("--file", type=str, required=True, help="Path to a .wav audio file")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"File not found: {args.file}")

    predict_emotion(args.file)
