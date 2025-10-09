# backend/app/schemas.py

from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class CameraBase(BaseModel):
    name: str
    rtsp_link: str
    coordinates: str

class CameraCreate(CameraBase):
    pass

class Camera(CameraBase):
    id: int

    class Config:
        from_attributes = True

class NumberPlateBase(BaseModel):
    camera_name: str
    plate_number: str
    image_path: Optional[str] = None
    # NEW: Optional status for flexibility
    status: Optional[str] = None

class NumberPlateCreate(NumberPlateBase):
    pass

class NumberPlate(NumberPlateBase):
    id: int
    created_at: datetime
    # Ensure status is included in the response schema
    status: str

    class Config:
        from_attributes = True
