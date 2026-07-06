"""
train.py
--------
End-to-end training pipeline:
1. Extract features from the dataset (feature_extraction.py)
2. Pad sequences to equal length, encode labels
3. Train/test split
4. Build and train the CNN+LSTM model (model.py)
5. Evaluate, plot confusion matrix, and save the model + label encoder
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical, pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from feature_extraction import build_dataset
from model import build_model

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

EPOCHS = 60
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    # 1. Extract features
    X, y = build_dataset()

    # 2. Pad all sequences to the same number of time frames
    max_len = max(sample.shape[0] for sample in X)
    X_padded = pad_sequences(
        [sample for sample in X],
        maxlen=max_len, dtype="float32", padding="post", truncating="post"
    )
    print(f"Padded feature shape: {X_padded.shape}")  # (num_samples, max_len, num_features)

    # 3. Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    num_classes = y_categorical.shape[1]
    print(f"Classes ({num_classes}): {list(label_encoder.classes_)}")

    # 4. Train/test split (stratified so rare classes are represented in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y_categorical, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y_encoded
    )

    # 5. Normalize features (per-feature standardization using train stats)
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Save normalization stats alongside the model (needed at inference time)
    np.save(os.path.join(MODELS_DIR, "norm_mean.npy"), mean)
    np.save(os.path.join(MODELS_DIR, "norm_std.npy"), std)

    # 6. Build model
    model = build_model(input_shape=X_train.shape[1:], num_classes=num_classes)
    model.summary()

    # 7. Train
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, "emotion_model.h5"),
            monitor="val_accuracy", save_best_only=True
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # 8. Evaluate
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(
        y_true, y_pred, target_names=label_encoder.classes_
    )
    print(report)

    with open(os.path.join(MODELS_DIR, "training_report.txt"), "w") as f:
        f.write("Speech Emotion Recognition — Training Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)

    # 9. Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Speech Emotion Recognition")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"))
    print(f"Confusion matrix saved to {MODELS_DIR}/confusion_matrix.png")

    # 10. Training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "training_curves.png"))
    print(f"Training curves saved to {MODELS_DIR}/training_curves.png")

    # 11. Save label encoder + max_len (needed by predict.py)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump({"label_encoder": label_encoder, "max_len": max_len}, f)

    print("\n✅ Training complete. Model + artifacts saved in models/")


if __name__ == "__main__":
    main()
