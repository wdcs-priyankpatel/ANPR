#!/usr/bin/env python3
"""
Simple ANPR Tracking Demo
- Input: RTSP stream + Region coordinates
- Output: Video with vehicle tracking, plate detection, and OCR results
- Shows: Track IDs, Vehicle Types, License Plates, Bounding Boxes
"""

import cv2
import os
import numpy as np
from ultralytics import YOLO, solutions
from paddleocr import PaddleOCR
import time
from datetime import datetime
import re

# ═══════════════════════════════════════════════════════════
# CONFIGURATION - EDIT THESE VALUES
# ═══════════════════════════════════════════════════════════

# RTSP Stream URL
RTSP_URL = "rtsp://admin:admin123@192.168.1.100:554/stream1"

# Region coordinates (x_min, y_min, x_max, y_max)
# Example: Monitor gate area from (100, 100) to (800, 600)
REGION_COORDS = {
    'x_min': 100,
    'y_min': 100,
    'x_max': 800,
    'y_max': 600
}

# Output video settings
OUTPUT_VIDEO_PATH = "anpr_tracking_output.mp4"
SHOW_LIVE_PREVIEW = True  # Set to False if running on headless server

# Model paths
PLATE_MODEL_PATH = "best.pt"  # Your custom plate detection model
VEHICLE_MODEL_PATH = "yolov8n.pt"  # Standard YOLOv8 for vehicles

# ═══════════════════════════════════════════════════════════
# VEHICLE CLASSES (COCO Dataset)
# ═══════════════════════════════════════════════════════════

VEHICLE_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# Colors for visualization (BGR format)
COLORS = {
    'vehicle_box': (0, 255, 0),      # Green for vehicle
    'plate_box': (0, 0, 255),        # Red for plate
    'region': (255, 255, 0),         # Cyan for monitoring region
    'text_bg': (0, 0, 0),            # Black background for text
    'text': (255, 255, 255),         # White text
}

# ═══════════════════════════════════════════════════════════
# INITIALIZE MODELS
# ═══════════════════════════════════════════════════════════

print("🚀 Initializing models...")

try:
    # Initialize OCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    print("✅ OCR initialized")
    
    # Initialize plate detection model
    plate_model = YOLO(PLATE_MODEL_PATH, task="detect")
    print(f"✅ Plate model loaded: {PLATE_MODEL_PATH}")
    
    # Initialize TrackZone for vehicle tracking
    region_points = [
        (REGION_COORDS['x_min'], REGION_COORDS['y_min']),  # Top-left
        (REGION_COORDS['x_max'], REGION_COORDS['y_min']),  # Top-right
        (REGION_COORDS['x_max'], REGION_COORDS['y_max']),  # Bottom-right
        (REGION_COORDS['x_min'], REGION_COORDS['y_max'])   # Bottom-left
    ]
    
    trackzone = solutions.TrackZone(
        show=False,
        region=region_points,
        model=VEHICLE_MODEL_PATH,
    )
    print(f"✅ TrackZone initialized with region: {region_points}")
    
except Exception as e:
    print(f"❌ Error initializing models: {e}")
    exit(1)

# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def clean_ocr_text(text):
    """Clean and validate OCR text"""
    # Remove spaces and special characters
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', text)
    
    # Remove common prefix errors
    cleaned = cleaned.lstrip('EF').replace("IND", "")
    
    # Validate against Indian license plate patterns
    pattern1 = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$", re.IGNORECASE)
    pattern2 = re.compile(r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$', re.IGNORECASE)
    pattern3 = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{2,3}[0-9]{1,4}$", re.IGNORECASE)
    
    if pattern1.match(cleaned) or pattern2.match(cleaned):
        return cleaned
    elif pattern3.match(cleaned):
        return cleaned
    
    return text.replace(" ", "")  # Return cleaned even if no pattern match

def is_plate_inside_vehicle(plate_box, vehicle_box):
    """Check if plate center is inside vehicle box"""
    plate_center_x = (plate_box[0] + plate_box[2]) / 2
    plate_center_y = (plate_box[1] + plate_box[3]) / 2
    
    return (
        plate_center_x > vehicle_box[0] and 
        plate_center_x < vehicle_box[2] and
        plate_center_y > vehicle_box[1] and 
        plate_center_y < vehicle_box[3]
    )

def draw_text_with_background(frame, text, position, font_scale=0.6, thickness=2):
    """Draw text with black background for better visibility"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    x, y = position
    
    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x, y - text_height - 5),
        (x + text_width + 5, y + 5),
        COLORS['text_bg'],
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        text,
        (x + 2, y),
        font,
        font_scale,
        COLORS['text'],
        thickness,
        cv2.LINE_AA
    )

def perform_ocr(frame, plate_box):
    """Perform OCR on plate region"""
    try:
        x1, y1, x2, y2 = [int(coord) for coord in plate_box]
        buffer = 15
        
        # Crop plate region with buffer
        y1_crop = max(0, y1 - buffer)
        y2_crop = min(frame.shape[0], y2 + buffer)
        x1_crop = max(0, x1 - buffer)
        x2_crop = min(frame.shape[1], x2 + buffer)
        
        cropped_plate = frame[y1_crop:y2_crop, x1_crop:x2_crop]
        
        if cropped_plate.shape[0] < 20 or cropped_plate.shape[1] < 20:
            return None, 0.0
        
        # Run OCR
        ocr_result = ocr.ocr(cropped_plate, cls=True)
        
        if ocr_result and ocr_result[0]:
            ocr_text = ""
            ocr_confidence = 0.0
            
            for line in ocr_result[0]:
                if line:
                    ocr_text += line[1][0] + " "
                    ocr_confidence += line[1][1]
            
            if ocr_result[0]:
                ocr_confidence /= len(ocr_result[0])
            
            cleaned_text = clean_ocr_text(ocr_text)
            
            return cleaned_text, ocr_confidence
    
    except Exception as e:
        print(f"OCR error: {e}")
    
    return None, 0.0

# ═══════════════════════════════════════════════════════════
# RTSP CONNECTION
# ═══════════════════════════════════════════════════════════

print(f"\n📡 Connecting to RTSP stream: {RTSP_URL}")

# Try multiple connection methods
cap = None

# Method 1: Try with FFmpeg
try:
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            print("✅ Connected via FFmpeg")
        else:
            cap.release()
            cap = None
except Exception as e:
    print(f"⚠️ FFmpeg connection failed: {e}")
    cap = None

# Method 2: Try with GStreamer
if cap is None:
    try:
        gst_pipeline = (
            f"rtspsrc location={RTSP_URL} latency=100 protocols=tcp "
            f"! rtph264depay ! h264parse ! avdec_h264 "
            f"! videoconvert ! appsink"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print("✅ Connected via GStreamer")
            else:
                cap.release()
                cap = None
    except Exception as e:
        print(f"⚠️ GStreamer connection failed: {e}")
        cap = None

# Method 3: Simple connection
if cap is None:
    try:
        cap = cv2.VideoCapture(RTSP_URL)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print("✅ Connected via default method")
            else:
                cap.release()
                cap = None
    except Exception as e:
        print(f"⚠️ Default connection failed: {e}")
        cap = None

if cap is None or not cap.isOpened():
    print("❌ Failed to connect to RTSP stream")
    exit(1)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

print(f"📐 Video properties: {frame_width}x{frame_height} @ {fps} FPS")

# ═══════════════════════════════════════════════════════════
# VIDEO WRITER SETUP
# ═══════════════════════════════════════════════════════════

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    print("❌ Failed to create output video file")
    exit(1)

print(f"💾 Output video: {OUTPUT_VIDEO_PATH}")

# ═══════════════════════════════════════════════════════════
# MAIN PROCESSING LOOP
# ═══════════════════════════════════════════════════════════

print("\n🎬 Starting video processing...")
print("Press 'q' to quit\n")

# Tracking data
plate_cache = {}  # vehicle_track_id -> {plate_number, confidence, last_seen}
frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("⚠️ Failed to read frame, retrying...")
            time.sleep(0.1)
            continue
        
        frame_count += 1
        display_frame = frame.copy()
        
        # ───────────────────────────────────────────────────
        # Draw monitoring region
        # ───────────────────────────────────────────────────
        cv2.rectangle(
            display_frame,
            (REGION_COORDS['x_min'], REGION_COORDS['y_min']),
            (REGION_COORDS['x_max'], REGION_COORDS['y_max']),
            COLORS['region'],
            3
        )
        
        # ───────────────────────────────────────────────────
        # STEP 1: Track vehicles in region using TrackZone
        # ───────────────────────────────────────────────────
        trackzone_result = trackzone(frame)
        
        current_vehicles = {}  # vehicle_track_id -> {box, vehicle_type}
        
        if hasattr(trackzone_result, 'boxes') and trackzone_result.boxes is not None:
            boxes = trackzone_result.boxes.xyxy.cpu().numpy()
            track_ids = trackzone_result.boxes.id
            classes = trackzone_result.boxes.cls.cpu().numpy()
            
            if track_ids is not None:
                track_ids = track_ids.cpu().numpy()
                
                for box, track_id, cls in zip(boxes, track_ids, classes):
                    vehicle_class_id = int(cls)
                    vehicle_track_id = int(track_id)
                    
                    # Only process recognized vehicle classes
                    if vehicle_class_id in VEHICLE_CLASSES:
                        vehicle_type = VEHICLE_CLASSES[vehicle_class_id]
                        vehicle_box = box.tolist()
                        
                        current_vehicles[vehicle_track_id] = {
                            'box': vehicle_box,
                            'vehicle_type': vehicle_type
                        }
        
        # ───────────────────────────────────────────────────
        # STEP 2: Detect license plates
        # ───────────────────────────────────────────────────
        plate_results = plate_model(frame, verbose=False)
        
        detected_plates = []
        
        if plate_results[0].boxes is not None:
            plate_boxes = plate_results[0].boxes.xyxy.cpu().numpy()
            plate_confidences = plate_results[0].boxes.conf.cpu().numpy()
            
            for box, confidence in zip(plate_boxes, plate_confidences):
                detected_plates.append({
                    'box': box.tolist(),
                    'confidence': float(confidence)
                })
        
        # ───────────────────────────────────────────────────
        # STEP 3: Associate plates with vehicles
        # ───────────────────────────────────────────────────
        for vehicle_id, vehicle_data in current_vehicles.items():
            vehicle_box = vehicle_data['box']
            vehicle_type = vehicle_data['vehicle_type']
            
            x1, y1, x2, y2 = [int(coord) for coord in vehicle_box]
            
            # Draw vehicle bounding box
            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                COLORS['vehicle_box'],
                2
            )
            
            # Prepare vehicle info text
            vehicle_info = f"ID:{vehicle_id} {vehicle_type}"
            plate_number = None
            plate_conf = 0.0
            
            # Check if this vehicle has an associated plate
            for plate_info in detected_plates:
                plate_box = plate_info['box']
                
                if is_plate_inside_vehicle(plate_box, vehicle_box):
                    # Draw plate bounding box
                    px1, py1, px2, py2 = [int(coord) for coord in plate_box]
                    cv2.rectangle(
                        display_frame,
                        (px1, py1),
                        (px2, py2),
                        COLORS['plate_box'],
                        2
                    )
                    
                    # Perform OCR (only once every 5 frames for performance)
                    if vehicle_id not in plate_cache or frame_count % 4 == 0:
                        ocr_text, ocr_conf = perform_ocr(frame, plate_box)
                        
                        if ocr_text and ocr_conf > 0.5:  # Confidence threshold
                            plate_cache[vehicle_id] = {
                                'plate_number': ocr_text,
                                'confidence': ocr_conf,
                                'last_seen': frame_count
                            }
                    
                    # Use cached plate number if available
                    if vehicle_id in plate_cache:
                        cached = plate_cache[vehicle_id]
                        if frame_count - cached['last_seen'] < 30:  # Valid for 30 frames
                            plate_number = cached['plate_number']
                            plate_conf = cached['confidence']
                    
                    break  # One plate per vehicle
            
            # Display vehicle info
            if plate_number:
                vehicle_info += f" | {plate_number}"
                vehicle_info += f" ({plate_conf:.2f})"
            
            draw_text_with_background(
                display_frame,
                vehicle_info,
                (x1, y1 - 10),
                font_scale=0.6,
                thickness=2
            )
        
        # ───────────────────────────────────────────────────
        # Display statistics
        # ───────────────────────────────────────────────────
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        stats = [
            f"Frame: {frame_count}",
            f"FPS: {current_fps:.1f}",
            f"Vehicles: {len(current_vehicles)}",
            f"Plates Detected: {len(detected_plates)}",
        ]
        
        y_offset = 30
        for stat in stats:
            draw_text_with_background(
                display_frame,
                stat,
                (10, y_offset),
                font_scale=0.7,
                thickness=2
            )
            y_offset += 35
        
        # ───────────────────────────────────────────────────
        # Write frame and display
        # ───────────────────────────────────────────────────
        out.write(display_frame)
        
        if SHOW_LIVE_PREVIEW:
            cv2.imshow('ANPR Tracking Demo', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⏹️ Stopped by user")
                break
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            print(f"📊 Processed {frame_count} frames | "
                  f"FPS: {current_fps:.1f} | "
                  f"Vehicles: {len(current_vehicles)} | "
                  f"Cached Plates: {len(plate_cache)}")

except KeyboardInterrupt:
    print("\n⏹️ Stopped by user (Ctrl+C)")

except Exception as e:
    print(f"\n❌ Error during processing: {e}")
    import traceback
    traceback.print_exc()

finally:
    # ═══════════════════════════════════════════════════════════
    # CLEANUP
    # ═══════════════════════════════════════════════════════════
    print("\n🧹 Cleaning up...")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Statistics:")
    print(f"   - Total frames: {frame_count}")
    print(f"   - Duration: {elapsed_time:.1f} seconds")
    print(f"   - Average FPS: {avg_fps:.1f}")
    print(f"   - Output video: {OUTPUT_VIDEO_PATH}")
    print(f"\n🎉 Done!")