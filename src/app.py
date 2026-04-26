from __future__ import annotations

import argparse
import logging

import cv2

from src.config import load_settings
from src.core.state_machine import ActivationState, GestureActivationFSM
from src.pipelines.gesture import HandGestureDetector, StableGestureEmitter
from src.utils.logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Driver fatigue detection skeleton")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--simulate-gestures",
        type=str,
        default="",
        help="Comma-separated test gestures, e.g. thumbs_up,peace_sign,ok_sign",
    )
    parser.add_argument("--video", type=str, default="", help="Path to video file. If empty, webcam is used.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--display", action="store_true", help="Show annotated video window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = full video)")
    parser.add_argument("--stable-frames", type=int, default=3, help="Frames required to emit a gesture event")
    parser.add_argument("--preferred-hand", type=str, default="right", choices=["right", "left", "any"])
    parser.add_argument("--gesture-debounce-sec", type=float, default=0.8)
    return parser.parse_args()


def _run_simulation(fsm: GestureActivationFSM, sequence: str, log: logging.Logger) -> None:
    gestures = [g.strip() for g in sequence.split(",") if g.strip()]
    if not gestures:
        log.info("Skeleton ready. Provide --simulate-gestures to test activation FSM.")
        snap = fsm.snapshot()
        log.info("State=%s next_expected=%s", snap.state.value, snap.next_expected)
        return

    ts = 0.0
    step = 1.0
    for gesture in gestures:
        snap = fsm.consume(gesture=gesture, ts=ts)
        log.info(
            "t=%.1f gesture=%s -> state=%s next_expected=%s matched=%d",
            ts,
            gesture,
            snap.state.value,
            snap.next_expected,
            snap.matched_count,
        )
        ts += step


def _run_live(args: argparse.Namespace, fsm: GestureActivationFSM, log: logging.Logger) -> None:
    source = args.video if args.video else args.camera_index
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    try:
        detector = HandGestureDetector(preferred_hand=args.preferred_hand)
    except RuntimeError as exc:
        raise RuntimeError(
            "Gesture detector initialization failed. Run in a local GUI-capable session "
            "or use --simulate-gestures for FSM-only checks."
        ) from exc
    emitter = StableGestureEmitter(min_stable_frames=args.stable_frames)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 25.0

    frame_idx = 0
    last_state = None
    last_emitted_label: str | None = None
    last_emitted_ts: float = -1e9

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if ts <= 0.0:
                ts = frame_idx / fps

            detection = detector.detect(frame)
            emitted = emitter.update(detection.label)
            if emitted is not None:
                if emitted == last_emitted_label and (ts - last_emitted_ts) < args.gesture_debounce_sec:
                    emitted = None
                else:
                    last_emitted_label = emitted
                    last_emitted_ts = ts

            snap = fsm.consume(gesture=emitted, ts=ts)

            if emitted is not None or snap.state != last_state:
                log.info(
                    "t=%.2f raw=%s(%.2f) emitted=%s state=%s next=%s matched=%d",
                    ts,
                    detection.label,
                    detection.confidence,
                    emitted,
                    snap.state.value,
                    snap.next_expected,
                    snap.matched_count,
                )
                last_state = snap.state

            if args.display:
                status_color = (0, 200, 0) if snap.state == ActivationState.ACTIVE else (0, 165, 255)
                cv2.putText(frame, f"state: {snap.state.value}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
                cv2.putText(
                    frame,
                    f"raw: {detection.label or '-'} ({detection.confidence:.2f})",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(frame, f"emitted: {emitted or '-'}", (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(
                    frame,
                    f"next: {snap.next_expected or '-'}",
                    (20, 141),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Driver Monitor - Gesture Activation", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        detector.close()
        cap.release()
        if args.display:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    configure_logging()
    log = logging.getLogger("app")

    settings = load_settings(args.config)
    fsm = GestureActivationFSM(settings.activation)

    if args.simulate_gestures:
        _run_simulation(fsm, args.simulate_gestures, log)
        return

    _run_live(args, fsm, log)


if __name__ == "__main__":
    main()
