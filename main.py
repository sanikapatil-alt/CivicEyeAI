
from pathlib import Path
from uuid import uuid4

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    UploadFile,
    File,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from database import engine, get_db, Base
from models import Complaint
from schemas import ComplaintCreate, ComplaintResponse
from ai_detection import extract_video_frames


# --------------------------------------------------
# 1. DATABASE INITIALIZATION
# --------------------------------------------------

Base.metadata.create_all(bind=engine)


# --------------------------------------------------
# 2. APPLICATION AND UPLOAD CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".mp4",
    ".mov",
}

VIDEO_EXTENSIONS = {".mp4", ".mov"}


app = FastAPI(
    title="CivicEye AI API",
    description="AI-powered civic complaint management system",
    version="1.0.0",
)


# --------------------------------------------------
# 3. STATIC FILE ACCESS
# --------------------------------------------------

app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOAD_DIR)),
    name="uploads",
)


# --------------------------------------------------
# 4. CORS CONFIGURATION
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 5. HOME AND HEALTH ENDPOINTS
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to CivicEye AI Backend!",
        "status": "Running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "CivicEye AI",
    }


# --------------------------------------------------
# 6. CIVIC ISSUE CATEGORIES
# --------------------------------------------------

@app.get("/api/issues")
def get_issue_categories():
    return {
        "categories": [
            "Road & Potholes",
            "Garbage & Waste",
            "Water Leakage",
            "Streetlights",
            "Road Signs",
            "Waterlogging",
            "Other",
        ]
    }


# --------------------------------------------------
# 7. CREATE A CIVIC COMPLAINT
# --------------------------------------------------

@app.post(
    "/api/complaints",
    response_model=ComplaintResponse,
)
def create_complaint(
    complaint: ComplaintCreate,
    db: Session = Depends(get_db),
):
    try:
        new_complaint = Complaint(
            tracking_id=f"CIV-{uuid4().hex[:8].upper()}",
            category=complaint.category,
            description=complaint.description,
            location=complaint.location,
            latitude=complaint.latitude,
            longitude=complaint.longitude,
            evidence_filename=complaint.evidence_filename,
        )

        db.add(new_complaint)
        db.commit()
        db.refresh(new_complaint)

        return new_complaint

    except SQLAlchemyError:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="Could not save complaint to database.",
        )


# --------------------------------------------------
# 8. GET ALL CIVIC COMPLAINTS
# --------------------------------------------------

@app.get(
    "/api/complaints",
    response_model=list[ComplaintResponse],
)
def get_complaints(
    db: Session = Depends(get_db),
):
    return (
        db.query(Complaint)
        .order_by(Complaint.created_at.desc())
        .all()
    )


# --------------------------------------------------
# 9. UPLOAD IMAGE OR VIDEO EVIDENCE
# --------------------------------------------------

@app.post("/api/uploads")
async def upload_evidence(
    file: UploadFile = File(...),
):
    original_filename = file.filename or "evidence"

    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        await file.close()

        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload JPG, JPEG, PNG, WEBP, MP4 or MOV.",
        )

    stored_filename = f"{uuid4().hex}{extension}"
    file_path = UPLOAD_DIR / stored_filename

    total_size = 0

    try:
        with file_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Maximum upload size is 100 MB.",
                    )

                destination.write(chunk)

    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise

    except OSError:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Could not save uploaded evidence.",
        )

    finally:
        await file.close()

    return {
        "message": "Evidence uploaded successfully",
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "file_url": f"/uploads/{stored_filename}",
        "file_type": (
            "video"
            if extension in VIDEO_EXTENSIONS
            else "image"
        ),
    }


# --------------------------------------------------
# 10. ANALYZE UPLOADED DASHCAM VIDEO
# --------------------------------------------------

@app.post("/api/analyze-video/{filename}")
def analyze_video(filename: str):
    # Prevent directory traversal and accept only video files.
    if Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    extension = Path(filename).suffix.lower()

    if extension not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Please provide an MP4 or MOV video.",
        )

    video_path = UPLOAD_DIR / filename

    if not video_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Video not found in uploads folder.",
        )

    try:
        result = extract_video_frames(video_path)

        if not isinstance(result, dict):
            raise ValueError(
                "Video processor returned an unexpected result."
            )

        required_keys = {
            "duration_seconds",
            "total_frames",
            "extracted_count",
            "frames",
        }

        if not required_keys.issubset(result):
            raise ValueError(
                "Video processor result is missing required fields."
            )

        return {
            "message": "Video processed successfully",
            "filename": filename,
            "duration_seconds": result["duration_seconds"],
            "total_frames": result["total_frames"],
            "extracted_count": result["extracted_count"],
            "frame_timestamps": [
                frame["timestamp_seconds"]
                for frame in result["frames"]
            ],
            "ai_detection_status": (
                "Frame extraction complete; "
                "pothole detection model integration pending."
            ),
        }

    except (ValueError, OSError, KeyError, TypeError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Video processing failed: {str(error)}",
        )


# --------------------------------------------------
# 11. APPLICATION STARTUP MESSAGE
# --------------------------------------------------

@app.get("/api/status")
def application_status():
    return {
        "application": "CivicEye AI",
        "backend": "Running",
        "database": "Configured",
        "upload_support": [
            "Images",
            "MP4 videos",
            "MOV videos",
        ],
        "video_processing": "Frame extraction enabled",
        "pothole_detection": "Model integration pending",
    }
