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

# Webcam + overlay window
python -m src.app --config configs/base.yaml --display
```

## Calibrate thumbs-up from your labeled data
If you have positive thumbs-up clips in `data/labels/thumbs_up`, generate personalized thresholds:
```bash
python -m src.tools.calibrate_thumbs_up --input-dir data/labels/thumbs_up --output models/thumbs_up_calibration.json --preferred-hand right
```
The live detector automatically loads `models/thumbs_up_calibration.json` if it exists.

## Project structure
- `configs/`: YAML configs
- `src/`: codebase
- `tests/`: unit tests
- `scripts/`: helper scripts
- `docs/`: report/supporting docs
- `data/raw`: captured videos
- `data/processed`, `data/labels`, `data/splits`: generated assets
