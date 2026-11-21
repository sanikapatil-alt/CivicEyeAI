import cv2
from pathlib import Path
from ultralytics import YOLO

# Project directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "ai_models" / "Pothole_best.pt"

# Load the trained pothole model once
model = YOLO(str(MODEL_PATH))


def extract_video_frames(video_path, interval_seconds=2):
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open the video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise ValueError("Could not determine video FPS.")

    duration_seconds = total_frames / fps

    # Create folders for extracted and annotated frames
    output_dir = UPLOAD_DIR / f"{video_path.stem}_frames"
    annotated_dir = output_dir / "annotated"

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    extracted_frames = []
    detections = []

    timestamp = 0
    frame_number = 0

    while timestamp <= duration_seconds:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        success, frame = cap.read()

        if not success:
            break

        frame_filename = f"frame_{timestamp:04d}.jpg"
        frame_path = output_dir / frame_filename

        cv2.imwrite(str(frame_path), frame)

        # Run pothole detection
        results = model.predict(
            source=frame,
            conf=0.25,
            classes=[0],
            verbose=False
        )

        result = results[0]
        annotated_frame = result.plot()

        annotated_filename = f"annotated_{timestamp:04d}.jpg"
        annotated_path = annotated_dir / annotated_filename

        cv2.imwrite(str(annotated_path), annotated_frame)

        frame_detections = []

        if result.boxes is not None:
            for box in result.boxes:
                confidence = float(box.conf[0])
                coordinates = box.xyxy[0].tolist()

                detection = {
                    "class": "pothole",
                    "confidence": round(confidence, 4),
                    "timestamp_seconds": timestamp,
                    "frame_filename": frame_filename,
                    "annotated_filename": annotated_filename,
                    "bounding_box": [
                        round(value, 2) for value in coordinates
                    ]
                }

                frame_detections.append(detection)
                detections.append(detection)

        extracted_frames.append({
            "timestamp_seconds": timestamp,
            "frame_filename": frame_filename,
            "frame_path": str(frame_path),
            "annotated_filename": annotated_filename,
            "annotated_path": str(annotated_path),
            "potholes_detected": len(frame_detections)
        })

        timestamp += interval_seconds
        frame_number += 1

    cap.release()

    return {
        "duration_seconds": round(duration_seconds, 2),
        "total_frames": total_frames,
        "extracted_count": len(extracted_frames),
        "frames": extracted_frames,
        "detected_count": len(detections),
        "detections": detections
    }