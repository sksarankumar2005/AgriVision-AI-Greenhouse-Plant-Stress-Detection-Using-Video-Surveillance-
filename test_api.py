"""Self-test: run before starting the web server."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def test_signatures():
    import inspect
    from predict import run_prediction

    params = inspect.signature(run_prediction).parameters
    required = {"input_path", "output_path", "conf"}
    missing = required - set(params.keys())
    if missing:
        raise AssertionError(f"run_prediction missing params: {missing}")


def test_legacy_call():
    from predict import run_prediction

    frame = ROOT / "static" / "test_frame.jpg"
    if not frame.exists():
        print("SKIP legacy test (no static/test_frame.jpg)")
        return

    result = run_prediction(
        str(frame),
        output_path=None,
        conf=0.25,
        max_seconds=None,
    )
    assert result.get("media_type") == "image"
    assert result["model_name"] == "best.pt"
    print("OK legacy call (output_path=None)")


def test_keyword_call():
    from predict import run_prediction

    clip = ROOT / "static" / "short_clip.mp4"
    if not clip.exists():
        print("SKIP video test (no static/short_clip.mp4)")
        return

    result = run_prediction(
        input_path=str(clip),
        conf=0.25,
        max_seconds=3.0,
        frame_stride=3,
    )
    assert result["media_type"] == "video"
    assert "summary" in result
    print(f"OK video call — {len(result['summary'])} class(es) in summary")


def test_positional_call():
    from predict import run_prediction

    frame = ROOT / "static" / "test_frame.jpg"
    if not frame.exists():
        return

    result = run_prediction(str(frame), None, conf=0.25)
    assert result["media_type"] == "image"
    print("OK positional call (input_path, output_path)")


if __name__ == "__main__":
    test_signatures()
    test_legacy_call()
    test_keyword_call()
    test_positional_call()
    print("All self-tests passed.")
