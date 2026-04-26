# Annotation Guide (Skeleton)

## Gesture labels
- `thumbs_up`
- `peace_sign`
- `ok_sign`
- `other`

## Fatigue labels
- `alert`
- `drowsy_eye`
- `yawn`
- `head_nod`

## Sequence rule
Activation sequence is strictly ordered:
1. `thumbs_up`
2. `peace_sign`
3. `ok_sign`

It must be completed within `sequence_timeout_sec` and each step must satisfy `max_gap_between_gestures_sec`.
