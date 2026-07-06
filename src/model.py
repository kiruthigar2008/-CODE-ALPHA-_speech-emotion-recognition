"""
model.py
--------
Defines the CNN + LSTM hybrid architecture used for speech emotion
recognition. Conv1D layers capture local spectral patterns per time
frame; LSTM layers capture how those patterns evolve over the utterance.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, MaxPooling1D, Dropout,
    LSTM, Dense, Input
)
from tensorflow.keras.optimizers import Adam


def build_model(input_shape: tuple, num_classes: int) -> Sequential:
    """
    Build the CNN+LSTM model.

    input_shape: (time_frames, num_features) — e.g. (130, 182)
    num_classes: number of emotion categories to classify
    """
    model = Sequential(name="Speech_Emotion_CNN_LSTM")

    model.add(Input(shape=input_shape))

    # --- CNN block 1 ---
    model.add(Conv1D(256, kernel_size=5, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # --- CNN block 2 ---
    model.add(Conv1D(128, kernel_size=5, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # --- LSTM block ---
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))

    # --- Dense classifier head ---
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    # Quick sanity check
    m = build_model(input_shape=(130, 182), num_classes=8)
    m.summary()
