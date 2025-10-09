# backend/app/crud.py

from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from . import models, schemas
import json
import os
from datetime import datetime, timedelta, timezone

# ------------------------------------------------------------------------------------------------
# NOTE on Datetime: Since you are likely using SQLite, it's safer to use naive datetimes 
# for comparison and storage. We will convert the UTC-aware time from the worker to naive.
# ------------------------------------------------------------------------------------------------

def get_number_plates(db: Session, skip: int = 0, limit: int = 20, camera_name: str = None, plate_number: str = None, start_date: str = None, end_date: str = None):
    # Ensure latest detections are first
    query = db.query(models.NumberPlate).order_by(desc(models.NumberPlate.created_at))
    
    if camera_name:
        query = query.filter(models.NumberPlate.camera_name == camera_name)

    if plate_number:
        query = query.filter(models.NumberPlate.plate_number.ilike(f"%{plate_number}%"))

    if start_date and end_date:
        start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
        query = query.filter(models.NumberPlate.created_at.between(start_datetime, end_datetime))

    return query.offset(skip).limit(limit).all()

# Function to get paginated data AND total count (updated sorting)
def get_paginated_number_plates(db: Session, skip: int = 0, limit: int = 20, camera_name: str = None, plate_number: str = None, start_date: str = None, end_date: str = None):
    query = db.query(models.NumberPlate)
    
    if camera_name:
        query = query.filter(models.NumberPlate.camera_name == camera_name)

    if plate_number:
        query = query.filter(models.NumberPlate.plate_number.ilike(f"%{plate_number}%"))

    if start_date and end_date:
        start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
        query = query.filter(models.NumberPlate.created_at.between(start_datetime, end_datetime))

    total_count = query.count()
    # Ensure latest detections are first
    paginated_data = query.order_by(desc(models.NumberPlate.created_at)).offset(skip).limit(limit).all()

    return {"data": paginated_data, "total_count": total_count}

def get_cameras(db: Session):
    return db.query(models.Camera).all()

def create_camera(db: Session, camera: schemas.CameraCreate):
    db_camera = models.Camera(
        name=camera.name,
        rtsp_link=camera.rtsp_link,
        coordinates=camera.coordinates
    )
    db.add(db_camera)
    db.commit()
    db.refresh(db_camera)
    return db_camera

def process_detection_event(db: Session, plate_data: dict):
    """
    Processes a new detection event, determines if it's an IN or OUT event 
    based on the 1-minute rule, and saves a new log entry.
    """
    
    current_time_aware = plate_data['created_at']
    # Convert UTC-aware time to naive time for SQLite storage/comparison
    current_time_naive = current_time_aware.replace(tzinfo=None)

    # 1. Get the most recent detection for this plate on this camera
    last_detection = db.query(models.NumberPlate).filter(
        models.NumberPlate.plate_number == plate_data['plate_number'],
        models.NumberPlate.camera_name == plate_data['camera_name']
    ).order_by(desc(models.NumberPlate.created_at)).first()
    
    new_status = "IN" # Default status is IN
    
    if last_detection:
        # Calculate time difference
        time_since_last = current_time_naive - last_detection.created_at
        
        # 1-minute (60 seconds) gap check
        if time_since_last.total_seconds() > 60:
            # If the last event was 'IN', the new event (after 1 min gap) is 'OUT'
            if last_detection.status == "IN":
                new_status = "OUT"
            # If the last event was 'OUT', the new event (after 1 min gap) is a new 'IN'
            else:
                new_status = "IN"
        
        else:
            # If the detection is within 1 minute, it's not a new IN/OUT event. Skip logging.
            return None # Signal to the worker to skip saving this log entry


    # 2. Create a new event record (IN or OUT)
    new_plate = models.NumberPlate(
        camera_name=plate_data['camera_name'],
        plate_number=plate_data['plate_number'],
        image_path=plate_data['image_path'],
        created_at=current_time_naive, 
        status=new_status
    )
    db.add(new_plate)
    db.commit()
    db.refresh(new_plate)
    return new_plate


def is_plate_recently_detected(db: Session, plate_number: str, camera_name: str, minutes: int = 5):
    """Checks if ANY log entry for this plate exists in the last 'minutes' to prevent over-logging."""
    # Use naive time for comparison
    time_limit = datetime.now() - timedelta(minutes=minutes) 
    
    return db.query(models.NumberPlate).filter(
        models.NumberPlate.plate_number == plate_number,
        models.NumberPlate.camera_name == camera_name,
        models.NumberPlate.created_at >= time_limit
    ).first() is not None

def get_camera_by_id(db: Session, camera_id: int):
    return db.query(models.Camera).filter(models.Camera.id == camera_id).first()
