from sqlalchemy import Column, Integer, String, DateTime
from .database import Base

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    rtsp_link = Column(String, unique=True, index=True)
    coordinates = Column(String)

class NumberPlate(Base):
    __tablename__ = "number_plates"

    id = Column(Integer, primary_key=True, index=True)
    camera_name = Column(String, index=True)
    plate_number = Column(String, index=True)
    image_path = Column(String)
    created_at = Column(DateTime, nullable=False)
    status = Column(String, default="IN")