import cv2
import os
import json
import numpy as np
from ultralytics import YOLO
import time
from paddleocr import PaddleOCR
import re
import csv
from collections import deque, defaultdict
from datetime import datetime, timedelta
import atexit
# Assuming these imports work correctly in your environment
from .database import SessionLocal
from . import crud, models
from sqlalchemy import desc
import subprocess
import threading
from queue import Queue, Empty

# Set environment variables for optimal Ubuntu performance
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '1'
os.environ['GST_DEBUG'] = '1'

# Define vehicle class mapping based on standard COCO/YOLO training
VEHICLE_CLASSES = {
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}

# Initialize global resources once per process
try:
    # Set up PaddleOCR to be quieter
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    
    # 1. Custom Plate Model (for OCR cropping)
    plate_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best.pt')
    plate_model = YOLO(plate_model_path, task="detect")
    
    # 2. General Vehicle Model with TRACKING enabled (YOLOv8n)
    vehicle_model = YOLO('yolov8s.pt')
    
except Exception as e:
    print(f"Error initializing models in anpr_worker: {e}")
    ocr, plate_model, vehicle_model = None, None, None

# Define file paths relative to the backend directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SAVE_DIR = os.path.join(BASE_DIR, "../detected_data")
STATUS_DIR = os.path.join(BASE_DIR, "../status")
VIDEO_FPS = 20
BUFFER_SIZE = int(VIDEO_FPS * 30)

os.makedirs(STATUS_DIR, exist_ok=True)

# Global dictionary to track processed detections
processed_detections = {}

def write_status(camera_id, status_message):
    """Writes the status of the worker to a file."""
    status_file = os.path.join(STATUS_DIR, f"{camera_id}.txt")
    with open(status_file, "w") as f:
        f.write(status_message)
    print(f"Status for camera {camera_id}: {status_message}")

def cleanup_status_file(camera_id):
    """Removes the status file on exit."""
    status_file = os.path.join(STATUS_DIR, f"{camera_id}.txt")
    if os.path.exists(status_file):
        os.remove(status_file)
    print(f"Cleaned up status file for camera {camera_id}.")

def is_frame_valid(frame):
    """Check if frame is valid and not corrupted"""
    if frame is None or frame.size == 0:
        return False
    if len(frame.shape) != 3 or frame.shape[0] < 100 or frame.shape[1] < 100:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)
    if variance < 50:
        return False
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total_pixels = frame.shape[0] * frame.shape[1]
    gray_range = hist[100:156].sum()
    if gray_range > 0.8 * total_pixels:
        return False
    return True

def test_rtsp_streams(base_rtsp_url):
    """Test different stream endpoints to find the best H.264 stream"""
    if '/stream' in base_rtsp_url:
        base_url = base_rtsp_url.split('/stream')[0]
    else:
        base_url = base_rtsp_url.rsplit('/', 1)[0] if '/' in base_rtsp_url else base_rtsp_url
    
    stream_paths = [
        '/h264',
        '/stream2',
        '/live/sub',
        '/live/main',
        '/video.h264',
        '/stream1',
    ]
    
    print(f"Testing RTSP streams for optimal H.264 compatibility...")
    
    for stream_path in stream_paths:
        test_url = base_url + stream_path
        print(f"Testing: {stream_path}")
        
        try:
            gst_pipeline = f"rtspsrc location={test_url} latency=100 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            
            if not cap.isOpened():
                cap = cv2.VideoCapture(test_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
            
            if cap.isOpened():
                good_frames = 0
                for _ in range(10):
                    ret, frame = cap.read()
                    if ret and is_frame_valid(frame):
                        good_frames += 1
                    time.sleep(0.1)
                
                cap.release()
                
                if good_frames >= 7:
                    print(f"✅ Found optimal stream: {stream_path} ({good_frames}/10 good frames)")
                    return test_url
                elif good_frames > 0:
                    print(f"⚠️ Partial success: {stream_path} ({good_frames}/10 good frames)")
                else:
                    print(f"❌ Poor quality: {stream_path}")
            else:
                print(f"❌ Cannot connect: {stream_path}")
                
        except Exception as e:
            print(f"❌ Error testing {stream_path}: {e}")
    
    print("No optimal H.264 stream found, using original URL")
    return base_rtsp_url

def create_optimized_capture(rtsp_url):
    """Create optimized video capture for Ubuntu with multiple backends"""
    
    print(f"Creating optimized capture for: {rtsp_url}")
    
    try:
        print("Attempting GStreamer pipeline...")
        gst_pipeline = (
            f"rtspsrc location={rtsp_url} "
            f"latency=100 "
            f"protocols=tcp "
            f"! rtph264depay "
            f"! h264parse "
            f"! avdec_h264 "
            f"! videoconvert "
            f"! video/x-raw,format=BGR "
            f"! appsink drop=1 max-buffers=1"
        )
        
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and is_frame_valid(test_frame):
                print("✅ GStreamer capture successful")
                return cap, "GStreamer"
        
        if cap: cap.release()
    except Exception as e:
        print(f"GStreamer failed: {e}")
    
    try:
        print("Attempting FFmpeg capture...")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and is_frame_valid(test_frame):
                print("✅ FFmpeg capture successful")
                return cap, "FFmpeg"
        
        if cap: cap.release()
    except Exception as e:
        print(f"FFmpeg failed: {e}")
    
    print("❌ All capture methods failed")
    return None, None

class RTSPFrameReader:
    """Threaded RTSP frame reader for better performance"""
    
    def __init__(self, rtsp_url, frame_queue):
        self.rtsp_url = rtsp_url
        self.frame_queue = frame_queue
        self.running = False
        self.cap = None
        self.backend = None
        self.thread = None
        
    def start(self):
        """Start the frame reading thread"""
        self.running = True
        self.thread = threading.Thread(target=self._read_frames)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop the frame reading thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        if self.cap:
            self.cap.release()
            
    def _read_frames(self):
        """Main frame reading loop"""
        consecutive_failures = 0
        max_failures = 20
        
        while self.running:
            try:
                if self.cap is None:
                    optimal_url = test_rtsp_streams(self.rtsp_url)
                    self.cap, self.backend = create_optimized_capture(optimal_url)
                    
                    if self.cap is None:
                        print("Failed to create capture, retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                    
                    print(f"Capture created with {self.backend} backend")
                    consecutive_failures = 0
                
                ret, frame = self.cap.read()
                
                if not ret or not is_frame_valid(frame):
                    consecutive_failures += 1
                    print(f"Frame read failure {consecutive_failures}/{max_failures}")
                    
                    if consecutive_failures >= max_failures:
                        print("Too many failures, recreating capture...")
                        if self.cap: self.cap.release()
                        self.cap = None
                        time.sleep(2)
                    
                    continue
                
                consecutive_failures = 0
                
                try:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            pass
                    
                    self.frame_queue.put(frame.copy(), block=False)
                    
                except Exception as e:
                    print(f"Queue error: {e}")
                
                time.sleep(0.033)
                
            except Exception as e:
                consecutive_failures += 1
                print(f"Frame reader error: {e}")
                
                if consecutive_failures >= max_failures:
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                    time.sleep(5)
        
        print("Frame reader thread stopped")

def read_coordinates_from_string(coordinates_string):
    try:
        coords = [int(c) for c in coordinates_string.split(',')]
        if len(coords) != 4:
            raise ValueError("Expected 4 comma-separated integers.")
        
        x_min, y_max, x_max, y_min = coords
        return (x_min, y_max), (x_max, y_min)
    except (ValueError, IndexError) as e:
        print(f"Error parsing coordinates: {e}. Using default coordinates.")
        return (100, 100), (300, 200)

def get_best_detection_for_track(ocr_results_list):
    if not ocr_results_list:
        return None
    
    best_idx = max(range(len(ocr_results_list)), 
                   key=lambda i: ocr_results_list[i]['combined_score'])
    return ocr_results_list[best_idx]

def get_most_frequent_text_from_best_detections(ocr_results_list):
    if not ocr_results_list:
        return ""
    
    sorted_results = sorted(ocr_results_list, key=lambda x: x['combined_score'], reverse=True)
    top_results_count = max(3, len(sorted_results) // 2)
    top_results = sorted_results[:top_results_count]
    
    text_list = [result['text'] for result in top_results if result['text']]
    cleaned_list = [re.sub(r'[^a-zA-Z0-9]', '', s).lstrip('EF') for s in text_list]
    
    pattern1 = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}", re.IGNORECASE)
    pattern2 = re.compile(r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}', re.IGNORECASE)
    pattern3 = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{2,3}[0-9]{1,4}", re.IGNORECASE)
    
    filtered_list = []
    filtered_list_pattern_3 = []
    
    for s in cleaned_list:  
        s = s.replace("IND", "")
        match1 = pattern1.match(s)
        match2 = pattern2.match(s)
        match3 = pattern3.match(s)
        if match1:
            filtered_list.append(match1.group(0))
        elif match2:
            filtered_list.append(match2.group(0))
        elif match3:
            filtered_list_pattern_3.append(match3.group(0))
    
    if not filtered_list:
        if filtered_list_pattern_3:
            unique, counts = np.unique(filtered_list_pattern_3, return_counts=True)
            index = np.argmax(counts)
            return unique[index]
        else:
            return ""
    else:
        unique, counts = np.unique(filtered_list, return_counts=True)
        index = np.argmax(counts)
        return unique[index]

def save_data_to_csv(csv_path, plate_data):
    headers = ['Timestamp', 'PlateNumber', 'VehicleType', 'DetectionScore', 'OcrConfidence', 'CombinedScore', 'Status']
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(plate_data)

def process_and_save_detection(track_id, active_tracks, camera_name, fps, frame_width, frame_height, frame_buffer, current_frame_for_full_img):
    """Process and save detection results, logging all tracked vehicles (with or without a plate)."""
    global processed_detections
    
    if track_id not in active_tracks:
        return

    track_data = active_tracks[track_id]
    
    # 1. Get plate and vehicle data
    most_freq = get_most_frequent_text_from_best_detections(track_data['plates'])
    best_plate_detection = get_best_detection_for_track(track_data['plates'])
    vehicle_type = track_data['vehicle_type']
    vehicle_bbox = track_data['vehicle_bbox']
    
    plate_tag = most_freq if most_freq else "NOPLATE"
    log_type = f"{'PLATE & ' if most_freq else ''}{vehicle_type.upper()}"
    
    timestamp = datetime.now()
    
    # 2. Check for overall cooldown
    db = SessionLocal()
    try:
        if most_freq and crud.is_plate_recently_detected(db, most_freq, camera_name, 5): 
            print(f"⏭️ Skipping {log_type} (Plate: {most_freq}), recently detected (5 min cooldown)")
            return
        elif not most_freq and crud.is_vehicle_recently_detected_only(db, vehicle_type, camera_name, 5):
             print(f"⏭️ Skipping {log_type} (Vehicle only), recently detected (5 min cooldown)")
             return
    finally:
        db.close()

    # --- DETERMINE PATHS ---
    date_str = timestamp.strftime('%Y-%m-%d')
    time_str = timestamp.strftime('%H-%M-%S')

    save_path_date = os.path.join(MAIN_SAVE_DIR, date_str)
    save_path_photo = os.path.join(save_path_date, 'plate_photo')
    save_path_full_frame = os.path.join(save_path_date, 'full_frame') 
    save_path_video = os.path.join(save_path_date, 'video')
    save_path_csv = os.path.join(save_path_date, 'csv')
    
    os.makedirs(save_path_photo, exist_ok=True)
    os.makedirs(save_path_full_frame, exist_ok=True)
    os.makedirs(save_path_video, exist_ok=True)
    os.makedirs(save_path_csv, exist_ok=True)

    # --- DETERMINE IN/OUT STATUS ---
    predicted_status = "DETECTED"
    
    if most_freq:
        db = SessionLocal()
        try:
            last_detection = db.query(models.NumberPlate).filter(
                models.NumberPlate.plate_number == most_freq,
                models.NumberPlate.camera_name == camera_name
            ).order_by(desc(models.NumberPlate.created_at)).first()
            
            predicted_status = "IN" 
            if last_detection:
                time_since_last = timestamp - last_detection.created_at
                if time_since_last.total_seconds() > 60:
                    predicted_status = "OUT" if last_detection.status == "IN" else "IN"
                else:
                    print(f"⏭️ Skipping {log_type}, too soon for IN/OUT event.")
                    return
        finally:
            db.close()

    # --- SAVE FULL FRAME IMAGE ---
    full_frame_name = f"{time_str}_{plate_tag}_{vehicle_type.upper()}_{predicted_status}.jpg"
    full_frame_path = os.path.join(save_path_full_frame, full_frame_name)
    
    v_x1, v_y1, v_x2, v_y2 = [int(i) for i in vehicle_bbox]
    frame_to_save = current_frame_for_full_img.copy() 
    cv2.rectangle(frame_to_save, (v_x1, v_y1), (v_x2, v_y2), (0, 255, 0), 2)
    cv2.putText(frame_to_save, f"{vehicle_type.upper()} ({plate_tag})", (v_x1, v_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imwrite(full_frame_path, frame_to_save)
    full_image_path_relative = os.path.join(date_str, 'full_frame', full_frame_name)

    # --- SAVE CROPPED PLATE PHOTO ---
    image_path_relative = None
    if best_plate_detection:
        photo_name = f"{time_str}_{most_freq}_{predicted_status}.jpg"
        photo_path = os.path.join(save_path_photo, photo_name)
        cv2.imwrite(photo_path, best_plate_detection['cropped_image'])
        image_path_relative = os.path.join(date_str, 'plate_photo', photo_name)

    # --- CREATE DB ENTRY ---
    db = SessionLocal()
    db_plate = None
    try:
        plate_data_for_db = {
            "camera_name": camera_name,
            "plate_number": most_freq if most_freq else None,
            "vehicle_type": vehicle_type,
            "image_path": image_path_relative,
            "full_image_path": full_image_path_relative, 
            "created_at": timestamp,
            "status": predicted_status
        }
        
        db_plate = crud.create_number_plate_log(db, plate_data_for_db)
        
        if db_plate is None:
            if os.path.exists(full_frame_path): os.remove(full_frame_path)
            if image_path_relative and os.path.exists(os.path.join(MAIN_SAVE_DIR, image_path_relative)): 
                os.remove(os.path.join(MAIN_SAVE_DIR, image_path_relative))
            return

    except Exception as e:
        print(f"❌ Database error during processing: {e}")
        if os.path.exists(full_frame_path): os.remove(full_frame_path)
        if image_path_relative and os.path.exists(os.path.join(MAIN_SAVE_DIR, image_path_relative)): 
            os.remove(os.path.join(MAIN_SAVE_DIR, image_path_relative))
        return
    finally:
        db.close()

    # --- SAVE VIDEO ---
    video_name = f"{time_str}_{plate_tag}_{vehicle_type.upper()}_{db_plate.status}.mp4"
    video_path = os.path.join(save_path_video, video_name)
    
    if frame_buffer and len(frame_buffer) > 0:
        try:
            actual_height, actual_width = None, None
            valid_frames = []
            
            for frame in frame_buffer:
                if frame is not None and frame.size > 0:
                    if actual_height is None:
                        actual_height, actual_width = frame.shape[:2]
                    
                    if actual_height and actual_width and frame.shape[:2] == (actual_height, actual_width):
                        valid_frames.append(frame)
                    elif actual_height and actual_width:
                        resized_frame = cv2.resize(frame, (actual_width, actual_height))
                        valid_frames.append(resized_frame)
            
            if valid_frames and actual_height and actual_width:
                codecs_to_try = [
                    ('mp4v', '.mp4'),
                    ('XVID', '.avi'),
                    ('X264', '.mp4'),
                    ('MJPG', '.avi'),
                ]
                
                video_writer = None
                final_video_path = None
                
                for codec, ext in codecs_to_try:
                    try:
                        temp_video_path = video_path.replace('.mp4', ext)
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        actual_fps = max(5, min(fps, 30))
                        
                        video_writer = cv2.VideoWriter(
                            temp_video_path, 
                            fourcc, 
                            actual_fps, 
                            (actual_width, actual_height)
                        )
                        
                        if video_writer.isOpened():
                            final_video_path = temp_video_path
                            print(f"✅ Video writer initialized with {codec} codec")
                            break
                        else:
                            if video_writer: video_writer.release()
                            video_writer = None
                            
                    except Exception as e:
                        print(f"❌ Failed to initialize {codec} codec: {e}")
                        if video_writer: video_writer.release()
                        video_writer = None
                
                if video_writer and video_writer.isOpened():
                    frames_written = 0
                    for frame in valid_frames:
                        try:
                            video_writer.write(frame)
                            frames_written += 1
                        except Exception as e:
                            print(f"Error writing frame: {e}")
                            break
                    
                    video_writer.release()
                    
                    if os.path.exists(final_video_path):
                        file_size = os.path.getsize(final_video_path)
                        if file_size > 1000:
                            print(f"✅ Video saved: {final_video_path} ({file_size} bytes, {frames_written} frames)")
                        else:
                            print(f"⚠️ Video file too small: {file_size} bytes")
                            try:
                                os.remove(final_video_path)
                            except:
                                pass
                    else:
                        print("❌ Video file was not created")
                else:
                    print("❌ Could not initialize any video codec")
                    
            else:
                print("❌ No valid frames found in buffer")
                
        except Exception as e:
            print(f"❌ Video saving error: {e}")

    # --- SAVE CSV ---
    csv_path = os.path.join(save_path_csv, f"{camera_name}_{date_str}.csv")
    csv_data = {
        'Timestamp': timestamp.strftime('%Y-%m-%d_%H-%M-%S'),
        'PlateNumber': plate_tag,
        'VehicleType': vehicle_type,
        'DetectionScore': round(best_plate_detection.get('detection_score', 0.0), 4) if best_plate_detection else 0.0,
        'OcrConfidence': round(best_plate_detection.get('confidence', 0.0), 4) if best_plate_detection else 0.0,
        'CombinedScore': round(best_plate_detection.get('combined_score', 0.0), 4) if best_plate_detection else 0.0,
        'Status': db_plate.status
    }
    save_data_to_csv(csv_path, csv_data)
    
    # --- UPDATE JSON ---
    full_json_path = os.path.join(BASE_DIR, "../results_json", f"{camera_name}.json")
    try:
        with open(full_json_path, 'r') as f:
            current_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        current_data = {}
        
    current_data[plate_tag] = {
        "detection_score": csv_data['DetectionScore'],
        "ocr_confidence": csv_data['OcrConfidence'],
        "combined_score": csv_data['CombinedScore'],
        "img_path": image_path_relative,
        "full_img_path": full_image_path_relative,
        "vehicle_type": vehicle_type,
        "status": db_plate.status
    }
    
    with open(full_json_path, 'w') as f:
        json.dump(current_data, f, indent=4)
    
    print(f"💾 Saved detection: {log_type} (Plate: {plate_tag}, Status: {db_plate.status})")
    processed_detections[track_id] = True

def anpr_worker(rtsp_link, output_json_path, coordinates_string, camera_name, camera_id):
    """Ubuntu-optimized ANPR worker with YOLOv8 built-in tracking (BoT-SORT/ByteTrack)."""
    atexit.register(cleanup_status_file, camera_id)
    
    frame_queue = Queue(maxsize=3)
    frame_reader = None
    frame_buffer = deque(maxlen=BUFFER_SIZE)
    
    actual_fps = VIDEO_FPS
    actual_width = 1920
    actual_height = 1080

    try:
        if ocr is None or plate_model is None or vehicle_model is None:
            print("Error: Models not initialized.")
            write_status(camera_id, "error")
            return

        full_output_json_path = os.path.join(BASE_DIR, "../results_json", output_json_path)
        os.makedirs(os.path.dirname(full_output_json_path), exist_ok=True)
        if not os.path.exists(full_output_json_path):
             with open(full_output_json_path, 'w') as f:
                 json.dump({}, f)

        rect_start, rect_end = read_coordinates_from_string(coordinates_string)
        x_min, y_min = min(rect_start[0], rect_end[0]), min(rect_start[1], rect_end[1])
        x_max, y_max = max(rect_start[0], rect_end[0]), max(rect_start[1], rect_end[1])
        
        # NOTE: Removed the fixed buffer variable here as it's now calculated dynamically below.
        
        print(f"🚀 Starting YOLO Tracking ANPR processing: {rtsp_link}")
        write_status(camera_id, "connecting")
        
        # Start threaded frame reader
        frame_reader = RTSPFrameReader(rtsp_link, frame_queue)
        frame_reader.start()
        
        # Wait for first frame
        print("Waiting for RTSP stream to stabilize...")
        wait_time = 0
        first_frame = None
        
        while wait_time < 30:
            try:
                first_frame = frame_queue.get(timeout=1.0)
                if first_frame is not None and first_frame.size > 0:
                    actual_height, actual_width = first_frame.shape[:2]
                    print(f"📐 Detected frame dimensions: {actual_width}x{actual_height}")
                    frame_queue.put(first_frame)
                    break
            except Empty:
                pass
            wait_time += 1
        
        if first_frame is None:
            write_status(camera_id, "error")
            print("❌ Timeout waiting for RTSP stream")
            return
            
        write_status(camera_id, "running")
        print(f"✅ Successfully connected to {rtsp_link}")

        # Active tracks: {track_id: {'vehicle_type': str, 'vehicle_bbox': list, 'plates': []}}
        active_tracks = {}
        previous_track_ids = set()
        frame_count = 0
        last_stats = time.time()
        fps_start_time = time.time()
        
        print("🎯 Using YOLOv8 Built-in Tracking (BoT-SORT)")
        
        while frame_reader.running:
            try:
                try:
                    frame = frame_queue.get(timeout=1.0)
                except Empty:
                    print("No frames received, checking connection...")
                    continue
                
                frame_count += 1
                frame_buffer.append(frame.copy())
                
                # Calculate actual FPS every 10 frames
                if frame_count % 10 == 0:
                    current_time = time.time()
                    elapsed = current_time - fps_start_time
                    if elapsed > 0:
                        actual_fps = 10 / elapsed
                    fps_start_time = current_time
                
                # Print stats every 30 seconds
                if time.time() - last_stats > 30:
                    fps_display = frame_count / (time.time() - last_stats + 30)
                    print(f"📊 Processing stats - Frames: {frame_count}, FPS: {fps_display:.1f}, Actual FPS: {actual_fps:.1f}, Active Tracks: {len(active_tracks)}")
                    last_stats = time.time()
                    frame_count = 0
                
                current_frame_for_save = frame.copy() 
                ocr_frame = frame.copy()
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                
                # --- 1. Vehicle Detection + Tracking (YOLOv8 with built-in tracker) ---
                # Using track() instead of predict() - this enables tracking
                vehicle_results = vehicle_model.track(
                    frame, 
                    persist=True,           # Persist tracks between frames
                    tracker="botsort.yaml", # Options: "botsort.yaml" or "bytetrack.yaml"
                    verbose=False,
                    classes=list(VEHICLE_CLASSES.keys())  # Only detect vehicles
                )
                
                tracked_vehicles_details = {}
                current_track_ids = set()
                
                if vehicle_results[0].boxes is not None and vehicle_results[0].boxes.id is not None:
                    boxes = vehicle_results[0].boxes.xyxy.cpu().numpy()
                    confidences = vehicle_results[0].boxes.conf.cpu().numpy()
                    classes = vehicle_results[0].boxes.cls.cpu().numpy()
                    track_ids = vehicle_results[0].boxes.id.cpu().numpy().astype(int)
                    
                    for box, confidence, cls_id, track_id in zip(boxes, confidences, classes, track_ids):
                        cls_id_int = int(cls_id)
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Check if vehicle is in detection zone
                        if cls_id_int in VEHICLE_CLASSES and x2 >= x_min and x1 <= x_max and y2 >= y_min and y1 <= y_max:
                            vehicle_type = VEHICLE_CLASSES.get(cls_id_int, "UNKNOWN")
                            
                            tracked_vehicles_details[track_id] = {
                                'box': [int(x1), int(y1), int(x2), int(y2)],
                                'type': vehicle_type,
                                'confidence': float(confidence)
                            }
                            
                            current_track_ids.add(track_id)

                # --- 2. Plate Detection (Custom Model) ---
                plate_detections = []
                plate_results = plate_model(frame, verbose=False) 
                
                if plate_results[0].boxes is not None:
                    p_boxes = plate_results[0].boxes.xyxy.cpu().numpy()
                    p_confidences = plate_results[0].boxes.conf.cpu().numpy()
                    
                    for p_box, p_conf in zip(p_boxes, p_confidences):
                        if p_conf > 0.5:
                            plate_detections.append({'box': p_box.astype(int).tolist(), 'conf': float(p_conf)})

                # --- 3. Association and OCR ---
                for track_id, detail in tracked_vehicles_details.items():
                    
                    # Initialize track if new
                    if track_id not in active_tracks:
                        active_tracks[track_id] = {
                            'vehicle_type': detail['type'],
                            'vehicle_bbox': detail['box'],
                            'plates': [],
                            'first_seen': frame_count
                        }
                        print(f"🆕 New Track ID: {track_id} ({detail['type'].upper()})")
                    else:
                        # Update vehicle bbox
                        active_tracks[track_id]['vehicle_bbox'] = detail['box']
                    
                    v_x1, v_y1, v_x2, v_y2 = detail['box']
                    
                    # Find plate within vehicle bounding box
                    for p_det in plate_detections:
                        p_x1, p_y1, p_x2, p_y2 = p_det['box']
                        p_conf = p_det['conf']
                        
                        # Check if plate is inside vehicle box
                        if p_x1 >= v_x1 and p_x2 <= v_x2 and p_y1 >= v_y1 and p_y2 <= v_y2:
                            
                            # **MODIFIED: Calculate Dynamic Buffer**
                            plate_width = p_x2 - p_x1
                            plate_height = p_y2 - p_y1
                            
                            # Use 20% of the smaller dimension as buffer, min 15, max 50
                            buffer = int(max(15, min(50, min(plate_width, plate_height) * 0.2)))
                            
                            # Perform OCR cropping using the dynamic buffer
                            y1_crop = max(0, p_y1 - buffer)
                            y2_crop = min(ocr_frame.shape[0], p_y2 + buffer)
                            x1_crop = max(0, p_x1 - buffer)
                            x2_crop = min(ocr_frame.shape[1], p_x2 + buffer)
                            
                            cropped_image = ocr_frame[y1_crop:y2_crop, x1_crop:x2_crop]
                            
                            # Original size check restriction remains
                            if cropped_image.shape[0] < 20 or cropped_image.shape[1] < 20:
                                continue
                                
                            ocr_result = ocr.ocr(cropped_image, cls=True)
                            ocr_res = ""
                            ocr_confidence = 0.0
                            
                            try:
                                if ocr_result and ocr_result[0]:
                                    for line in ocr_result[0]:
                                        if line:
                                            ocr_res += line[1][0] + " "
                                            ocr_confidence += line[1][1]
                                    if ocr_result[0]:
                                        ocr_confidence /= len(ocr_result[0])
                            except Exception as e:
                                pass
                            
                            active_tracks[track_id]['plates'].append({
                                'text': ocr_res.replace(" ", ""),
                                'confidence': ocr_confidence,
                                'detection_score': p_conf,
                                'combined_score': (ocr_confidence + p_conf) / 2,
                                'cropped_image': cropped_image.copy(),
                                'frame_num': frame_count
                            })
                            
                            break  # Only process one plate per vehicle per frame

                # --- 4. Process Tracks that have left the scene ---
                leaving_ids = previous_track_ids - current_track_ids
                
                for leaving_id in leaving_ids:
                    if leaving_id in active_tracks:
                        frames_tracked = frame_count - active_tracks[leaving_id]['first_seen']
                        print(f"🚪 Track {leaving_id} leaving (tracked for {frames_tracked} frames)")
                        
                        if leaving_id not in processed_detections:
                            process_and_save_detection(
                                leaving_id, 
                                active_tracks, 
                                camera_name, 
                                actual_fps, 
                                actual_width, 
                                actual_height, 
                                list(frame_buffer),
                                current_frame_for_save
                            )
                        
                        del active_tracks[leaving_id]
                
                # Update previous tracks for next iteration
                previous_track_ids = current_track_ids.copy()

            except Exception as e:
                print(f"Error processing frame: {e}")
                import traceback
                traceback.print_exc()
                continue

    except Exception as e:
        write_status(camera_id, "error")
        print(f"❌ Error in anpr_worker for {rtsp_link}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if frame_reader:
            frame_reader.stop()
        print(f"🏁 Stopped processing: {rtsp_link}")