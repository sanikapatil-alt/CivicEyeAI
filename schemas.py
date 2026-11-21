
from pydantic import BaseModel, ConfigDict
from typing import Optional


class ComplaintCreate(BaseModel):
    category: str
    description: str
    location: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    evidence_filename: Optional[str] = None


class ComplaintResponse(BaseModel):
    id: int
    tracking_id: str
    category: str
    description: str
    location: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    evidence_filename: Optional[str] = None
    department: str
    status: str

    model_config = ConfigDict(from_attributes=True)
