"""
Greenhouse plant stress detection using trained YOLO weights (best.pt).
Returns detection summaries only — no annotated media files for the web app.
"""
import argparse
import uuid
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_MODEL = PROJECT_ROOT / "yolo26n.pt"
TRAINED_MODEL = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"

DISPLAY_NAMES = {
    "healthy": "Healthy Leaf",
    "wilting": "Wilting",
    "yellowing": "Yellowing",
    "tip_burn": "Tip-burn",
    "dry_leaf": "Dry Leaf",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def resolve_path(raw_path: str | Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def resolve_model_path(weights: str | Path | None = None) -> Path:
    if weights:
        path = resolve_path(weights)
        if not path.exists():
            raise FileNotFoundError(f"Model weights not found at {path}")
        return path
    if TRAINED_MODEL.exists():
        return TRAINED_MODEL
    if DEFAULT_BASE_MODEL.exists():
        return DEFAULT_BASE_MODEL
    raise FileNotFoundError(
        "No model weights found. Train with train.py or place yolo26n.pt in the project root."
    )


def format_label(class_name: str) -> str:
    key = class_name.lower().replace(" ", "_").replace("-", "_")
    return DISPLAY_NAMES.get(key, class_name.replace("_", " ").title())


def box_to_dict(box, names: dict[int, str]) -> dict[str, Any]:
    cls_id = int(box.cls[0])
    raw_name = names[cls_id]
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
    return {
        "class_id": cls_id,
        "class": raw_name,
        "label": format_label(raw_name),
        "confidence": round(confidence, 4),
        "confidence_pct": round(confidence * 100, 1),
        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        "is_stress": raw_name.lower() != "healthy",
    }


def summarize_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_class: dict[str, dict[str, Any]] = {}
    for det in detections:
        key = det["class"]
        if key not in by_class:
            by_class[key] = {
                "class": key,
                "label": det["label"],
                "count": 0,
                "max_confidence": 0.0,
                "max_confidence_pct": 0.0,
                "is_stress": det["is_stress"],
            }
        entry = by_class[key]
        entry["count"] += 1
        if det["confidence"] > entry["max_confidence"]:
            entry["max_confidence"] = det["confidence"]
            entry["max_confidence_pct"] = det["confidence_pct"]
    summary = list(by_class.values())
    summary.sort(key=lambda item: (-int(item["is_stress"]), -item["max_confidence"]))
    return summary


def load_yolo(weights: str | Path | None = None) -> YOLO:
    path = resolve_model_path(weights)
    print(f"[AgriVision] Using model: {path.name}")
    return YOLO(str(path))


def infer_frame(yolo: YOLO, source, conf: float, imgsz: int = 640):
    """Run YOLO; if nothing found, retry once at lower confidence."""
    results = yolo(source, conf=conf, imgsz=imgsz, verbose=False)[0]
    if len(results.boxes) == 0 and conf > 0.08:
        fallback_conf = min(conf, 0.12)
        results = yolo(source, conf=fallback_conf, imgsz=imgsz, verbose=False)[0]
        if len(results.boxes) > 0:
            return results, fallback_conf, True
    return results, conf, False


def predict_image(
    *,
    image_path: Path,
    yolo: YOLO,
    conf: float = 0.25,
    save_path: Path | None = None,
) -> dict[str, Any]:
    image_path = resolve_path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    results, used_conf, retried = infer_frame(yolo, str(image_path), conf, imgsz=640)

    if save_path:
        save_path = resolve_path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), results.plot())

    detections = [box_to_dict(box, yolo.names) for box in results.boxes]
    summary = summarize_detections(detections)

    return {
        "media_type": "image",
        "input_path": str(image_path),
        "detections": detections,
        "summary": summary,
        "detection_count": len(detections),
        "has_stress": any(item["is_stress"] for item in summary),
        "conf_used": used_conf,
        "conf_retried": retried,
    }


def predict_video(
    *,
    video_path: Path,
    yolo: YOLO,
    conf: float = 0.15,
    timeline_interval: int = 50,
    frame_stride: int = 5,
    imgsz: int = 416,
    max_seconds: float | None = 30.0,
) -> dict[str, Any]:
    video_path = resolve_path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frame_stride = max(1, int(frame_stride))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0 or fps <= 0:
        cap.release()
        raise ValueError("Could not read video properties. The file may be corrupt.")

    max_frames = int(fps * max_seconds) if max_seconds and max_seconds > 0 else None
    truncated = bool(max_frames and total_frames > max_frames)

    all_detections: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    frame_count = 0
    inference_count = 0

    while cap.isOpened():
        if max_frames and frame_count >= max_frames:
            break

        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count != 1 and frame_count % frame_stride != 0:
            continue

        inference_count += 1
        results, _, _ = infer_frame(yolo, frame, conf, imgsz=imgsz)
        frame_detections = [box_to_dict(box, yolo.names) for box in results.boxes]
        all_detections.extend(frame_detections)

        if frame_count == 1 or frame_count % timeline_interval == 0:
            timeline.append(
                {
                    "frame": frame_count,
                    "detections": frame_detections,
                    "summary": summarize_detections(frame_detections),
                    "status": _frame_status(frame_detections),
                }
            )

    cap.release()
    summary = summarize_detections(all_detections)

    return {
        "media_type": "video",
        "input_path": str(video_path),
        "detections": all_detections,
        "summary": summary,
        "timeline": timeline,
        "detection_count": len(all_detections),
        "total_frames": total_frames,
        "processed_frames": frame_count,
        "inference_frames": inference_count,
        "frame_stride": frame_stride,
        "fps": round(fps, 2),
        "has_stress": any(item["is_stress"] for item in summary),
        "truncated": truncated,
        "analyzed_seconds": round(frame_count / fps, 1) if fps else 0,
    }


def _frame_status(detections: list[dict[str, Any]]) -> str:
    stress = [d for d in detections if d["is_stress"]]
    if not stress and not detections:
        return "No plant regions detected"
    if not stress:
        return "Healthy"
    top = max(stress, key=lambda d: d["confidence"])
    return f"{top['label']} detected"


def run_prediction(
    input_path: str | Path,
    output_path: str | Path | None = None,
    weights: str | Path | None = None,
    model_path: str | Path | None = None,
    conf: float = 0.25,
    frame_stride: int = 5,
    imgsz: int = 416,
    max_seconds: float | None = 30.0,
    save_path: str | Path | None = None,
    save_annotated_video: bool | None = None,
    **_: Any,
) -> dict[str, Any]:
    """
    Run stress detection on an image or video.

    Legacy kwargs (output_path, model_path, save_annotated_video) are accepted
    for compatibility with older app versions and ignored when not needed.
    """
    del save_annotated_video

    if model_path is not None and weights is None:
        weights = model_path
    if output_path is not None and save_path is None:
        save_path = output_path

    input_path = resolve_path(input_path)
    suffix = input_path.suffix.lower()
    yolo = load_yolo(weights)
    weights_file = resolve_model_path(weights)

    if suffix in IMAGE_EXTENSIONS:
        result = predict_image(
            image_path=input_path,
            yolo=yolo,
            conf=conf,
            save_path=resolve_path(save_path) if save_path else None,
        )
    elif suffix in VIDEO_EXTENSIONS:
        result = predict_video(
            video_path=input_path,
            yolo=yolo,
            conf=conf,
            frame_stride=frame_stride,
            imgsz=imgsz,
            max_seconds=max_seconds,
        )
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    result["model_path"] = str(weights_file)
    result["model_name"] = weights_file.name
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Greenhouse plant stress detection")
    parser.add_argument("--input", "-i", required=True, help="Input image or video path")
    parser.add_argument("--output", "-o", help="Optional annotated image output (images only)")
    parser.add_argument("--weights", "-w", default=None, help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_prediction(
        args.input,
        output_path=args.output,
        weights=args.weights,
        conf=args.conf,
        max_seconds=None,
    )

    print("-------------------------------------------------")
    print(f"Model: {result['model_name']}")
    print(f"Media: {result['media_type']}")
    print(f"Regions: {result['detection_count']}")
    for item in result["summary"]:
        tag = "STRESS" if item["is_stress"] else "OK"
        print(f"  [{tag}] {item['label']} — {item['max_confidence_pct']}% (x{item['count']})")
    print("-------------------------------------------------")


if __name__ == "__main__":
    main()
