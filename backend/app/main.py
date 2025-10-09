import os
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import Dict, Optional
from starlette.responses import JSONResponse

from . import models, crud, database
from . import detection_manager as dm

app = FastAPI()

# IMPORTANT: Ensure the static file mount reflects the new save structure or adjust paths
# Mounting the parent directory of the save structure
app.mount("/images", StaticFiles(directory="detected_data"), name="images") 

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    models.Base.metadata.create_all(bind=database.engine)
    print("Database tables created.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the ANPR API"}

@app.get("/cameras")
def get_cameras(db: Session = Depends(database.get_db)):
    cameras = crud.get_cameras(db)
    for camera in cameras:
        camera.status = dm.get_status(camera.id)
    return cameras

@app.post("/cameras")
def create_camera(camera: dict, db: Session = Depends(database.get_db)):
    if not all(k in camera for k in ["name", "rtsp_link", "coordinates"]):
        raise HTTPException(status_code=400, detail="Invalid camera data")
    
    if db.query(models.Camera).filter(models.Camera.rtsp_link == camera["rtsp_link"]).first():
        raise HTTPException(status_code=409, detail="Camera with this RTSP link already exists")
    
    db_camera = models.Camera(
        name=camera["name"], 
        rtsp_link=camera["rtsp_link"], 
        coordinates=camera["coordinates"]
    )
    db.add(db_camera)
    db.commit()
    db.refresh(db_camera)
    return db_camera

@app.put("/cameras/{camera_id}")
def update_camera(camera_id: int, camera_data: Dict, db: Session = Depends(database.get_db)):
    db_camera = crud.get_camera_by_id(db, camera_id)
    if not db_camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    for field, value in camera_data.items():
        if hasattr(db_camera, field):
            setattr(db_camera, field, value)

    db.commit()
    db.refresh(db_camera)
    return db_camera

@app.post("/start-detection/{camera_id}")
def start_detection(camera_id: int, db: Session = Depends(database.get_db)):
    camera = crud.get_camera_by_id(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    output_json_filename = f"{camera.name}.json"
    
    started, message = dm.start_detection(
        camera_id=camera.id, 
        rtsp_link=camera.rtsp_link, 
        output_json_path=output_json_filename,
        coordinates=camera.coordinates,
        camera_name=camera.name
    )
    if not started:
        raise HTTPException(status_code=409, detail=message)
    return {"message": message, "status": "starting"}

@app.post("/stop-detection/{camera_id}")
def stop_detection(camera_id: int):
    stopped, message = dm.stop_detection(camera_id)
    if not stopped:
        raise HTTPException(status_code=409, detail=message)
    return {"message": message, "status": "stopped"}

@app.get("/detection-status/{camera_id}")
def get_detection_status(camera_id: int):
    status_file = os.path.join(os.path.dirname(__file__), f"../status/{camera_id}.txt")
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            status = f.read().strip()
            return {"status": status}
    return {"status": "stopped"}

@app.get("/number-plates")
def get_number_plates(
    skip: int = 0,
    limit: int = 20,
    camera_name: Optional[str] = None,
    plate_number: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(database.get_db)
):
    result = crud.get_paginated_number_plates(db, skip=skip, limit=limit, camera_name=camera_name, plate_number=plate_number, start_date=start_date, end_date=end_date)
    # The sorting is now handled in crud.py (descending order by created_at)
    return {"data": result["data"], "total_count": result["total_count"]}

@app.get("/number-plates/image/{image_path:path}")
def get_number_plate_image(image_path: str):
    # Adjust path to match the new StaticFiles mount point
    full_path = f"detected_data/{image_path}" 
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(full_path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
