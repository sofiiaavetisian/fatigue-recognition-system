from src.core.state_machine import ActivationState, GestureActivationFSM
from src.core.types import ActivationConfig


def _fsm() -> GestureActivationFSM:
    cfg = ActivationConfig(
        sequence=["thumbs_up", "peace_sign", "ok_sign"],
        sequence_timeout_sec=10.0,
        max_gap_between_gestures_sec=5.0,
        cooldown_sec=1.0,
    )
    return GestureActivationFSM(cfg)


def test_correct_sequence_activates():
    fsm = _fsm()
    fsm.consume("thumbs_up", 0.0)
    fsm.consume("peace_sign", 1.0)
    snap = fsm.consume("ok_sign", 2.0)
    assert snap.state == ActivationState.ACTIVE


def test_active_state_persists():
    fsm = _fsm()
    fsm.consume("thumbs_up", 0.0)
    fsm.consume("peace_sign", 1.0)
    fsm.consume("ok_sign", 2.0)
    snap = fsm.consume(None, 3.0)
    assert snap.state == ActivationState.ACTIVE


def test_wrong_order_resets():
    fsm = _fsm()
    fsm.consume("thumbs_up", 0.0)
    snap = fsm.consume("ok_sign", 1.0)
    assert snap.state == ActivationState.LISTENING
    assert snap.matched_count == 0
    assert snap.next_expected == "thumbs_up"


def test_timeout_resets():
    fsm = _fsm()
    fsm.consume("thumbs_up", 0.0)
    snap = fsm.consume("peace_sign", 11.0)
    assert snap.state == ActivationState.LISTENING
    assert snap.matched_count == 0


def test_duplicate_previous_gesture_is_ignored():
    fsm = _fsm()
    fsm.consume("thumbs_up", 0.0)
    # Duplicate thumbs_up should not reset progress.
    snap = fsm.consume("thumbs_up", 0.4)
    assert snap.state == ActivationState.LISTENING
    assert snap.matched_count == 1
    assert snap.next_expected == "peace_sign"
