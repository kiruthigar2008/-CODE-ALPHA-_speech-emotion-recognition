# 📥 Dataset Setup

This project is configured by default for the **RAVDESS** dataset, but works with **TESS** or **EMO-DB** with small filename-parsing tweaks in `src/feature_extraction.py`.

## Option 1: RAVDESS (recommended, default)

1. Download `Audio_Speech_Actors_01-24.zip` from Zenodo:
   https://zenodo.org/record/1188976
2. Extract it here so the structure looks like:
   ```
   data/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav
   data/RAVDESS/Actor_02/03-01-05-02-01-01-02.wav
   ...
   ```
3. RAVDESS filenames encode the emotion label in the **3rd number**:

   | Code | Emotion |
   |---|---|
   | 01 | neutral |
   | 02 | calm |
   | 03 | happy |
   | 04 | sad |
   | 05 | angry |
   | 06 | fearful |
   | 07 | disgust |
   | 08 | surprised |

   `feature_extraction.py` parses this automatically.

## Option 2: TESS

1. Download from: https://tspace.library.utoronto.ca/handle/1807/24487
2. Extract into `data/TESS/`. TESS folders are already named by emotion
   (e.g. `OAF_angry/`, `YAF_happy/`), so update `DATASET_TYPE = "TESS"`
   in `src/feature_extraction.py` — it will read the emotion from the
   folder name instead of the filename.

## Option 3: EMO-DB

1. Download from: http://emodb.bilderbar.info/download/
2. Extract into `data/EMO-DB/`. EMO-DB encodes emotion as a single letter
   in the filename (e.g. `03a01Fa.wav` → `F` = happy). Set
   `DATASET_TYPE = "EMODB"` in `src/feature_extraction.py`.

## Note

Audio data is **not included in this repo** (see `.gitignore`) since RAVDESS
alone is ~1GB+. Download it locally before running `train.py`.
