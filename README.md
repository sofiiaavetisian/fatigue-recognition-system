# fatigue-recognition-system

Driver fatigue detection with gesture-based activation.

## Activation sequence
The system activates only when this ordered gesture sequence is detected:
1. `thumbs_up`
2. `peace_sign`
3. `ok_sign`

## The Hybrid Brain
This system uses a "Dual-Brain" approach to maximize reliability:

*   **Classical (The Math):** Uses MediaPipe Face Mesh to compute **Eye Aspect Ratio (EAR)**, **Mouth Aspect Ratio (MAR)**, and **head pitch in degrees** via `cv2.solvePnP`. These features feed a **scikit-learn classifier** (RandomForest / LogisticRegression) trained on auto-labeled frames; if no classifier is available, the detector falls back to a hand-tuned rule-based score.
*   **Modern (The AI):** A **MobileNetV3-Small** CNN. Inference runs on a **face crop** (extracted from the same Face Mesh output) — not the whole frame — so the model focuses on the eyes/mouth/head region.
*   **Weighted Fusion:** The final score is a weighted average (**55% Math / 45% AI**), ensuring the system works even if the AI is uncertain or the lighting is poor for geometric math.

## Reproducible setup

### Option 1: conda
```bash
conda env create -f environment.yml
conda activate fatigue-recognition-system
```

### Option 2: venv
```bash
make setup
source .venv/bin/activate
```

## Testing & Verification
Before running the full app, verify the core logic and hardware compatibility:
```bash
# Run the full test suite (FSM, Hand Landmarks, UI)
python3 -m unittest discover tests

# Specifically verify the Hybrid weight math (0.55/0.45 split)
python3 -m tests.test_hybrid
```

## Quick checks
```bash
make test
python -m src.app --config configs/base.yaml
python -m src.app --config configs/base.yaml --simulate-gestures thumbs_up,peace_sign,ok_sign
```

## Run with real video/webcam
```bash
# Video file
python -m src.app --config configs/base.yaml --video data/raw/IMG_8221.MOV --max-frames 400

> **Note for Mac Users:** This system is configured to run AI inference on the **CPU** by default. This ensures stability on Intel/AMD MacBooks and avoids common crashes associated with the MPS (Metal) backend.

# Rotate video (if recorded upside-down or sideways)
# Options: 90 (clockwise), 180 (flip), 270 (counter-clockwise)
python -m src.app --config configs/base.yaml --video data/raw/IMG_8234.MOV --rotate 180 --display

# Webcam + overlay window
python -m src.app --config configs/base.yaml --display
```

## Calibrate thumbs-up from your labeled data
If you have positive thumbs-up clips in `data/labels/thumbs_up`, generate personalized thresholds:
```bash
python -m src.tools.calibrate_thumbs_up --input-dir data/labels/thumbs_up --output models/thumbs_up_calibration.json --preferred-hand right
```
The live detector automatically loads `models/thumbs_up_calibration.json` if it exists.

## Modern Pipeline Data Preparation
To train or run the AI-based fatigue detector, you must generate a labeled dataset from the raw videos.

1. **Verify Orientations**: 
   Ensure all videos are upright (press `r` to rotate, `s` to save, `q` to quit):
   ```bash
   python -m src.tools.check_orientations
   ```
2. **Generate Labeled Dataset**:
    Run the preprocessing script. This uses the Classical Pipeline to auto-label frames and organizes them into train/val/test splits:
    ```bash
   python -m src.tools.preprocess_videos
   ```
   Note: This script populates data/processed and data/splits using settings from configs/rotation_map.json.

## Modern Model Training & Evaluation
Once the dataset is prepared and split, we can train and verify the Deep Learning fatigue model (MobileNetV3).

### How to Run
1. **Train the modern model**: `python -m src.tools.train_modern`
2. **Train the classical classifier**: `python -m src.tools.train_classical_classifier`
3. **Evaluate modern accuracy**: `python -m src.tools.evaluate_modern`
4. **Visual demo**: `python -m src.tools.test_modern_model --video data/raw/IMG_8221.MOV`

### Artifacts
- Modern model weights: `models/fatigue_model.pt`
- Modern training metrics: `models/training_summary.json`
- Classical classifier: `models/classical_classifier.pkl`
- Classical classifier metrics: `models/classical_classifier_summary.json`

> **Note:** The shipped `fatigue_model.pt` was trained on full frames. The
> current pipeline now feeds the model **face crops** (which is the right
> thing to do). For best results, re-run `preprocess_videos` and
> `train_modern` so the model sees the same input distribution at training
> and inference time.

## Project structure
- `configs/`: YAML configs
- `src/`: codebase
- `tests/`: unit tests
- `models/`: trained AI models (fatigue_model.pt) and calibration files
- `scripts/`: helper scripts
- `docs/`: report/supporting docs
- `data/raw`: captured videos
- `data/processed`, `data/labels`, `data/splits`: generated assets
