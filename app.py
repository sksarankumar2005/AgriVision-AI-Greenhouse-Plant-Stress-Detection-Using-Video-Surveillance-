"""
AgriVision AI web server — greenhouse plant stress detection.
"""
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Force reload of predict module (no stale bytecode from old signatures)
import predict as predict_module
from predict import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, run_prediction

APP_VERSION = "2.0.0"
predict_module.__dict__["__version__"] = APP_VERSION

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "static"
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
MAX_CONTENT_LENGTH = 200 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Confirm the correct server build is running."""
    import inspect

    sig = inspect.signature(run_prediction)
    return jsonify(
        {
            "status": "ok",
            "app_version": APP_VERSION,
            "predict_file": str(Path(predict_module.__file__).resolve()),
            "accepts_output_path": "output_path" in sig.parameters,
            "model": "runs/detect/train/weights/best.pt",
        }
    )


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        abort(400, description="No file uploaded")

    file = request.files["file"]
    if not file or file.filename == "":
        abort(400, description="No file selected")

    if not allowed_file(file.filename):
        abort(400, description="Supported: JPG, PNG, WEBP, MP4, AVI, MOV, MKV")

    conf = request.form.get("conf", type=float, default=0.15)
    conf = max(0.05, min(conf, 0.95))
    frame_stride = request.form.get("frame_stride", type=int, default=5)
    frame_stride = max(1, min(frame_stride, 10))

    safe_name = secure_filename(file.filename) or "upload"
    token = uuid.uuid4().hex[:8]
    input_path = OUTPUT_DIR / f"input_{token}_{safe_name}"
    file.save(str(input_path))

    is_video = Path(safe_name).suffix.lower() in VIDEO_EXTENSIONS

    try:
        # Works with current and legacy call styles
        result = run_prediction(
            input_path=str(input_path),
            output_path=None,
            conf=conf,
            frame_stride=frame_stride if is_video else 1,
            max_seconds=30.0 if is_video else None,
        )
    except FileNotFoundError as exc:
        abort(404, description=str(exc))
    except ValueError as exc:
        abort(400, description=str(exc))
    except TypeError as exc:
        abort(
            500,
            description=(
                f"{exc}. Stop the server (Ctrl+C), run start.ps1, then try again."
            ),
        )
    except Exception as exc:
        abort(500, description=f"Inference failed: {exc}")

    response = {
        "success": True,
        "media_type": result["media_type"],
        "filename": safe_name,
        "summary": result["summary"],
        "detection_count": result["detection_count"],
        "has_stress": result["has_stress"],
        "conf_threshold": conf,
        "model_name": result.get("model_name", "best.pt"),
    }

    if result["media_type"] == "image":
        response["detections"] = result["detections"]
        response["conf_used"] = result.get("conf_used", conf)
        response["conf_retried"] = result.get("conf_retried", False)
    else:
        response["timeline"] = result.get("timeline", [])
        response["total_frames"] = result.get("total_frames", 0)
        response["processed_frames"] = result.get("processed_frames", 0)
        response["inference_frames"] = result.get("inference_frames", 0)
        response["frame_stride"] = result.get("frame_stride", frame_stride)
        response["fps"] = result.get("fps", 0)
        response["truncated"] = result.get("truncated", False)
        response["analyzed_seconds"] = result.get("analyzed_seconds", 0)

    return jsonify(response)


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(500)
def handle_error(error):
    description = getattr(error, "description", str(error))
    code = getattr(error, "code", 500)
    return jsonify({"success": False, "error": description}), code


if __name__ == "__main__":
    print(f"[AgriVision] Starting server v{APP_VERSION}")
    print(f"[AgriVision] predict.py -> {Path(predict_module.__file__).resolve()}")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
