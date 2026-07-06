# 🎙️ Speech Emotion Recognition (SER)

Recognize human emotions — happy, sad, angry, calm, fearful, disgust, surprised, neutral — directly from raw speech audio using deep learning and audio signal processing.

> Built as part of the **CodeAlpha Machine Learning Internship — Task 2: Emotion Recognition from Speech**

---

## 📌 Objective

Given a short audio clip of someone speaking, predict the emotional state of the speaker using acoustic features extracted from the speech signal.

## 🧠 Approach

1. **Signal Processing** — extract MFCCs (Mel-Frequency Cepstral Coefficients) along with supporting features (Chroma, Mel Spectrogram, Zero Crossing Rate, Root Mean Square Energy) from each audio clip using `librosa`.
2. **Deep Learning Model** — a hybrid **CNN + LSTM** architecture:
   - Conv1D layers learn local spectral patterns across the MFCC frames.
   - LSTM layers learn the temporal/sequential dependencies of how emotion evolves across the utterance.
   - Dense + Softmax layers classify the final emotion.
3. **Training** — the model is trained on the **RAVDESS** dataset (Ryerson Audio-Visual Database of Emotional Speech and Song), with support for **TESS** and **EMO-DB** as drop-in alternatives.

## 🗂️ Datasets Supported

| Dataset | Description | Link |
|---|---|---|
| **RAVDESS** | 24 actors, 8 emotions, studio quality | [zenodo.org/record/1188976](https://zenodo.org/record/1188976) |
| **TESS** | 2 actresses, 7 emotions | [tspace.library.utoronto.ca](https://tspace.library.utoronto.ca/handle/1807/24487) |
| **EMO-DB** | German speech emotion database, 7 emotions | [emodb.bilderbar.info](http://emodb.bilderbar.info/download/) |

By default this project is configured for **RAVDESS** filename conventions. See `data/README.md` for setup instructions and how to switch datasets.

## 🏗️ Project Structure

```
speech-emotion-recognition/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md          # dataset download + folder setup instructions
├── src/
│   ├── feature_extraction.py   # MFCC / Chroma / Mel / ZCR / RMS extraction
│   ├── model.py                 # CNN + LSTM model architecture
│   ├── train.py                 # training pipeline (extract -> train -> save)
│   └── predict.py               # run inference on a single audio file
├── models/                 # saved trained model + label encoder (generated after training)
└── notebooks/
    └── exploration.ipynb   # optional EDA / waveform & spectrogram visualization
```

## ⚙️ Setup

```bash
git clone <your-repo-url>
cd speech-emotion-recognition
pip install -r requirements.txt
```

## 📥 Dataset Setup

See [`data/README.md`](data/README.md). In short:

1. Download RAVDESS (`Audio_Speech_Actors_01-24.zip`) from Zenodo.
2. Extract it into `data/RAVDESS/` so you get folders like `data/RAVDESS/Actor_01/*.wav`.

## 🚀 Usage

**1. Train the model**
```bash
python src/train.py
```
This will:
- Walk through `data/RAVDESS/`, extract MFCC + Chroma + Mel + ZCR + RMS features per file
- Split into train/test sets
- Train the CNN+LSTM model
- Save the trained model to `models/emotion_model.h5`
- Save the label encoder to `models/label_encoder.pkl`
- Print accuracy, classification report, and confusion matrix

**2. Predict emotion from a new audio file**
```bash
python src/predict.py --file path/to/your_audio.wav
```
Output:
```
🎧 Predicted Emotion: happy (confidence: 91.4%)
```

## 🧪 Model Architecture

```
Input (MFCC + Chroma + Mel + ZCR + RMS features, reshaped to sequence)
    ↓
Conv1D (256 filters) + BatchNorm + ReLU + MaxPool + Dropout
    ↓
Conv1D (128 filters) + BatchNorm + ReLU + MaxPool + Dropout
    ↓
LSTM (128 units, return_sequences=True)
    ↓
LSTM (64 units)
    ↓
Dense (64, ReLU) + Dropout
    ↓
Dense (num_emotions, Softmax)
```

## 📊 Emotions Classified (RAVDESS)

`neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`

## 🛠️ Tech Stack

- **Python 3.9+**
- **librosa** — audio loading & feature extraction
- **TensorFlow / Keras** — CNN + LSTM model
- **scikit-learn** — train/test split, label encoding, metrics
- **numpy / pandas** — data handling
- **matplotlib / seaborn** — visualization (confusion matrix, waveforms)

## 📈 Results

After training on the full RAVDESS dataset (~1440 samples, 8 classes), the model typically reaches **~70-85% test accuracy** depending on train/test split, number of epochs, and data augmentation used. Exact numbers will print at the end of `train.py` and are saved to `models/training_report.txt`.

## 🔮 Future Improvements

- Add data augmentation (pitch shift, time stretch, noise injection) to improve generalization
- Try transformer-based / wav2vec2 embeddings instead of hand-crafted MFCCs
- Combine RAVDESS + TESS + EMO-DB for a larger, more diverse training set
- Deploy as a simple Flask/Streamlit web app for live mic-based emotion detection

## 📄 License

MIT License — free to use for academic and internship purposes.

---
*Part of the CodeAlpha ML Internship — Task 2*
