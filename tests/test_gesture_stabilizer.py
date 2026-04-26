from src.pipelines.gesture import StableGestureEmitter


def test_emits_once_after_min_stable_frames():
    emitter = StableGestureEmitter(min_stable_frames=3)
    assert emitter.update("thumbs_up") is None
    assert emitter.update("thumbs_up") is None
    assert emitter.update("thumbs_up") == "thumbs_up"
    assert emitter.update("thumbs_up") is None


def test_resets_on_none_and_label_change():
    emitter = StableGestureEmitter(min_stable_frames=2)
    assert emitter.update("peace_sign") is None
    assert emitter.update("peace_sign") == "peace_sign"
    assert emitter.update(None) is None
    assert emitter.update("ok_sign") is None
    assert emitter.update("ok_sign") == "ok_sign"
