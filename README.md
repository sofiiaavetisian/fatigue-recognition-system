# fatigue-recognition-system

Driver fatigue detection with gesture-based activation.

## Activation sequence
The system activates only when this ordered gesture sequence is detected:
1. `thumbs_up`
2. `peace_sign`
3. `ok_sign`

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
1. **Train the model**: `python -m src.tools.train_modern`
2. **Evaluate accuracy**: `python -m src.tools.evaluate_modern`
3. **Visual Demo**: `python -m src.tools.test_modern_model`

### Artifacts
- Model Weights: `models/fatigue_model.pt`
- Training Metrics: `models/training_summary.json`

## Project structure
- `configs/`: YAML configs
- `src/`: codebase
- `tests/`: unit tests
- `models/`: trained AI models (fatigue_model.pt) and calibration files
- `scripts/`: helper scripts
- `docs/`: report/supporting docs
- `data/raw`: captured videos
- `data/processed`, `data/labels`, `data/splits`: generated assets
