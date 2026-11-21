
from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy.sql import func

from database import Base


class Complaint(Base):
    __tablename__ = "complaints"

    id = Column(Integer, primary_key=True, index=True)

    tracking_id = Column(String(30), unique=True, index=True, nullable=False)

    category = Column(String(100), nullable=False)

    description = Column(Text, nullable=False)

    location = Column(String(500), nullable=False)

    latitude = Column(Float, nullable=True)

    longitude = Column(Float, nullable=True)

    evidence_filename = Column(String(255), nullable=True)

    department = Column(String(100), default="Pending Assignment")

    status = Column(String(50), default="Submitted")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
