"""
feature_extraction.py
----------------------
Extracts acoustic features (MFCC, Chroma, Mel Spectrogram, Zero Crossing
Rate, RMS Energy) from speech audio files for emotion recognition.

Supports RAVDESS (default), TESS, and EMO-DB dataset layouts.
"""

import os
import glob
import numpy as np
import librosa
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG — change this if you're using a different dataset
# ---------------------------------------------------------------------------
DATASET_TYPE = "RAVDESS"   # "RAVDESS" | "TESS" | "EMODB"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", DATASET_TYPE)

SAMPLE_RATE = 22050
DURATION = 3.0          # seconds — clips are padded/truncated to this length
N_MFCC = 40
N_MELS = 128

# RAVDESS emotion code -> label
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# EMO-DB letter code -> label
EMODB_EMOTION_MAP = {
    "W": "angry",
    "L": "boredom",
    "E": "disgust",
    "A": "fearful",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}


def get_label_from_path(filepath: str) -> str:
    """Extract the emotion label from a file path based on DATASET_TYPE."""
    filename = os.path.basename(filepath)

    if DATASET_TYPE == "RAVDESS":
        parts = filename.split("-")
        code = parts[2]  # 3rd segment is the emotion code
        return RAVDESS_EMOTION_MAP.get(code, "unknown")

    elif DATASET_TYPE == "TESS":
        # TESS folders are named like "OAF_angry" or "YAF_happy"
        parent_folder = os.path.basename(os.path.dirname(filepath))
        for emo in ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
                    "pleasant_surprise"]:
            if emo in parent_folder.lower():
                return "fearful" if emo == "fear" else ("surprised" if "surprise" in emo else emo)
        return "unknown"

    elif DATASET_TYPE == "EMODB":
        code = filename[5]  # 6th character encodes emotion
        return EMODB_EMOTION_MAP.get(code, "unknown")

    else:
        raise ValueError(f"Unknown DATASET_TYPE: {DATASET_TYPE}")


def extract_features(file_path: str) -> np.ndarray:
    """
    Load an audio file and extract a fixed-length feature sequence:
    MFCC (40) + Chroma (12) + Mel (128) + ZCR (1) + RMS (1) = 182 features
    per frame, stacked across time frames.
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    # Pad if shorter than expected duration
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    # Stack all features along the frequency axis -> shape (features, time_frames)
    combined = np.vstack([mfcc, chroma, librosa.power_to_db(mel), zcr, rms])

    # Transpose to (time_frames, features) so LSTM reads it as a sequence
    return combined.T


def build_dataset():
    """
    Walk through DATA_DIR, extract features + labels for every audio file.
    Returns X (list of 2D feature arrays) and y (list of string labels).
    """
    file_paths = glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)

    if not file_paths:
        raise FileNotFoundError(
            f"No .wav files found in {DATA_DIR}. "
            f"See data/README.md for dataset download instructions."
        )

    X, y = [], []
    print(f"Found {len(file_paths)} audio files. Extracting features...")

    for path in tqdm(file_paths):
        try:
            features = extract_features(path)
            label = get_label_from_path(path)
            if label == "unknown":
                continue
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"⚠️  Skipping {path}: {e}")

    return X, y


if __name__ == "__main__":
    X, y = build_dataset()
    print(f"Extracted {len(X)} samples.")
    print(f"Feature shape per sample: {X[0].shape}")
    print(f"Unique labels: {sorted(set(y))}")
