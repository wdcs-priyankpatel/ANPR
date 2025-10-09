# backend/app/anpr_worker.py

import cv2
import os
import json
import numpy as np
from ultralytics import YOLO
import time
from paddleocr import PaddleOCR
import re
from .sort.sort import Sort
import csv
from collections import deque
from datetime import datetime, timedelta # Removed timezone
import atexit
from .database import SessionLocal
from . import crud
import subprocess
import threading
from queue import Queue, Empty

# Set environment variables for optimal Ubuntu performance
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '1'
os.environ['GST_DEBUG'] = '1' # Reduce GStreamer debug output

# Initialize global resources once per process
try:
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    mot_tracker = Sort()
    # Ensure this path is correct relative to where the worker executes
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '/home/admin2/priyank/01_number_plate/backend/app/best.pt')
    model = YOLO(model_path, task="detect")
except Exception as e:
    print(f"Error initializing models in anpr_worker: {e}")
    ocr, mot_tracker, model = None, None, None

# Define file paths relative to the backend directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SAVE_DIR = os.path.join(BASE_DIR, "../detected_data")
STATUS_DIR = os.path.join(BASE_DIR, "../status")
VIDEO_FPS = 20
BUFFER_SIZE = int(VIDEO_FPS * 30)

os.makedirs(STATUS_DIR, exist_ok=True)

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
    
    # Check dimensions
    if len(frame.shape) != 3 or frame.shape[0] < 100 or frame.shape[1] < 100:
        return False
    
    # Check for gray/corrupted frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)
    
    # If variance is too low, likely corrupted
    if variance < 50:
        return False
    
    # Check histogram for uniform grayness
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total_pixels = frame.shape[0] * frame.shape[1]
    gray_range = hist[100:156].sum()
    
    # If more than 80% pixels are gray, likely corrupted
    if gray_range > 0.8 * total_pixels:
        return False
    
    return True

def test_rtsp_streams(base_rtsp_url):
    """Test different stream endpoints to find the best H.264 stream"""
    if '/stream' in base_rtsp_url:
        base_url = base_rtsp_url.split('/stream')[0]
    else:
        base_url = base_rtsp_url.rsplit('/', 1)[0] if '/' in base_rtsp_url else base_rtsp_url
    
    # Common H.264 stream paths
    stream_paths = [
        '/h264',           # Direct H.264
        '/stream2',        # Secondary stream (usually H.264)
        '/live/sub',       # Sub stream (lower quality H.264)
        '/live/main',      # Main stream
        '/video.h264',     # Some cameras
        '/stream1',        # Original path
    ]
    
    print(f"Testing RTSP streams for optimal H.264 compatibility...")
    
    for stream_path in stream_paths:
        test_url = base_url + stream_path
        print(f"Testing: {stream_path}")
        
        try:
            # Test with GStreamer pipeline first (Ubuntu native)
            gst_pipeline = f"rtspsrc location={test_url} latency=100 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            
            if not cap.isOpened():
                # Fallback to FFmpeg
                cap = cv2.VideoCapture(test_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
            
            if cap.isOpened():
                # Test frame reading
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
    
    # Try GStreamer pipeline first (best for Ubuntu)
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
            # Test frame reading
            ret, test_frame = cap.read()
            if ret and is_frame_valid(test_frame):
                print("✅ GStreamer capture successful")
                return cap, "GStreamer"
        
        if cap: cap.release()
    except Exception as e:
        print(f"GStreamer failed: {e}")
    
    # Try FFmpeg with optimizations
    try:
        print("Attempting FFmpeg capture...")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        # Optimized settings
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
                # Create capture if needed
                if self.cap is None:
                    optimal_url = test_rtsp_streams(self.rtsp_url)
                    self.cap, self.backend = create_optimized_capture(optimal_url)
                    
                    if self.cap is None:
                        print("Failed to create capture, retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                    
                    print(f"Capture created with {self.backend} backend")
                    consecutive_failures = 0
                
                # Read frame
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
                
                # Frame is good
                consecutive_failures = 0
                
                # Add to queue (non-blocking)
                try:
                    if self.frame_queue.full():
                        # Remove oldest frame
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            pass
                    
                    self.frame_queue.put(frame.copy(), block=False)
                    
                except Exception as e:
                    print(f"Queue error: {e}")
                
                time.sleep(0.033)  # ~30 FPS max
                
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

def missing_elements(list1, list2):
    return list(set(list1) - set(list2))

def save_data_to_csv(csv_path, plate_data):
    headers = ['Timestamp', 'PlateNumber', 'DetectionScore', 'OcrConfidence', 'CombinedScore', 'Status']
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(plate_data)

def process_and_save_detection(track_id, active_tracks, camera_name, fps, frame_width, frame_height, frame_buffer):
    """Process and save detection results with improved video handling"""
    global processed_detections
    
    if track_id not in active_tracks:
        return

    most_freq = get_most_frequent_text_from_best_detections(active_tracks[track_id])
    best_detection = get_best_detection_for_track(active_tracks[track_id])
    
    # 1. Use datetime.datetime.now() to get a naive datetime object (per user request)
    timestamp = datetime.now()
    
    if most_freq and best_detection:
        
        # 2. Check for overall cooldown to prevent over-logging (5 minute check)
        db = SessionLocal()
        try:
            # NOTE: crud.is_plate_recently_detected must be compatible with naive datetimes
            if crud.is_plate_recently_detected(db, most_freq, camera_name, 5): 
                 print(f"⏭️ Skipping {most_freq}, recently detected (within 5 min cooldown)")
                 return
        finally:
            db.close()

        # Re-initialize DB session for final save operation
        db = SessionLocal()
        
        # 3. Save to database and determine IN/OUT status
        plate_data_for_db = {
            "camera_name": camera_name,
            "plate_number": most_freq,
            "image_path": "", # Placeholder, path determined after saving
            "created_at": timestamp 
        }

        db_plate = None
        try:
            # Call the new processing function
            # NOTE: crud.process_detection_event must now handle a naive datetime object
            db_plate = crud.process_detection_event(db, plate_data_for_db)
            
            # If the crud function returns None, it means the event was skipped (gap < 1 min)
            if db_plate is None:
                 print(f"⏭️ Skipping {most_freq}, too soon for IN/OUT event.")
                 return

        except Exception as e:
            print(f"❌ Database error during IN/OUT processing: {e}")
            return
        finally:
            db.close()

        # --- File Saving (Only proceeds if a DB entry was successfully created) ---
        
        # Determine paths based on timestamp
        date_str = timestamp.strftime('%Y-%m-%d')
        save_path_date = os.path.join(MAIN_SAVE_DIR, date_str)
        save_path_photo = os.path.join(save_path_date, 'photo')
        save_path_video = os.path.join(save_path_date, 'video')
        save_path_csv = os.path.join(save_path_date, 'csv')
        
        os.makedirs(save_path_photo, exist_ok=True)
        os.makedirs(save_path_video, exist_ok=True)
        os.makedirs(save_path_csv, exist_ok=True)

        # Save photo (Adding status to filename)
        photo_name = f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}_{most_freq}_{db_plate.status}.jpg"
        photo_path = os.path.join(save_path_photo, photo_name)
        cv2.imwrite(photo_path, best_detection['cropped_image'])
        
        # Get relative path for database update
        image_path_relative = os.path.join(date_str, 'photo', photo_name)
        
        # Update image path in the database record we just created
        db = SessionLocal()
        try:
            db_plate.image_path = image_path_relative
            db.commit()
        except Exception as e:
            print(f"❌ Failed to update image path in DB: {e}")
        finally:
            db.close()
            
        # IMPROVED VIDEO SAVING
        video_name = f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}_{most_freq}_{db_plate.status}.mp4"
        video_path = os.path.join(save_path_video, video_name)
        
        if frame_buffer and len(frame_buffer) > 0:
            try:
                # Get actual frame dimensions from the first valid frame
                actual_height, actual_width = None, None
                valid_frames = []
                
                for frame in frame_buffer:
                    if frame is not None and frame.size > 0:
                        if actual_height is None:
                            actual_height, actual_width = frame.shape[:2]
                        
                        # Ensure all frames have the same dimensions
                        if actual_height and actual_width and frame.shape[:2] == (actual_height, actual_width):
                            valid_frames.append(frame)
                        elif actual_height and actual_width:
                            # Resize frame to match the expected dimensions
                            resized_frame = cv2.resize(frame, (actual_width, actual_height))
                            valid_frames.append(resized_frame)
                
                if valid_frames and actual_height and actual_width:
                    # Try multiple codecs for better compatibility
                    codecs_to_try = [
                        ('mp4v', '.mp4'),  # MP4V codec
                        ('XVID', '.avi'),  # XVID codec
                        ('X264', '.mp4'),  # H264 codec
                        ('MJPG', '.avi'),  # Motion JPEG
                    ]
                    
                    video_writer = None
                    final_video_path = None
                    
                    for codec, ext in codecs_to_try:
                        try:
                            temp_video_path = video_path.replace('.mp4', ext)
                            fourcc = cv2.VideoWriter_fourcc(*codec)
                            
                            # Use actual FPS or default to 10 if too high/low
                            actual_fps = max(5, min(fps, 30))  # Clamp between 5-30 FPS
                            
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
                        # Write frames to video
                        frames_written = 0
                        for frame in valid_frames:
                            try:
                                video_writer.write(frame)
                                frames_written += 1
                            except Exception as e:
                                print(f"Error writing frame: {e}")
                                break
                        
                        video_writer.release()
                        
                        # Verify the video was created successfully
                        if os.path.exists(final_video_path):
                            file_size = os.path.getsize(final_video_path)
                            if file_size > 1000:  # At least 1KB
                                print(f"✅ Video saved: {final_video_path} ({file_size} bytes, {frames_written} frames)")
                            else:
                                print(f"⚠️ Video file too small: {file_size} bytes")
                                # Try to remove the invalid file
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

        # Save CSV (Adding status to CSV data)
        csv_path = os.path.join(save_path_csv, f"{camera_name}_{date_str}.csv")
        csv_data = {
            'Timestamp': timestamp.strftime('%Y-%m-%d_%H-%M-%S'), 
            'PlateNumber': most_freq,
            'DetectionScore': round(best_detection['detection_score'], 4),
            'OcrConfidence': round(best_detection['confidence'], 4),
            'CombinedScore': round(best_detection['combined_score'], 4),
            'Status': db_plate.status
        }
        save_data_to_csv(csv_path, csv_data)
        
        # Update JSON (Adding status to JSON data)
        full_json_path = os.path.join(BASE_DIR, "../results_json", f"{camera_name}.json")
        try:
            with open(full_json_path, 'r') as f:
                current_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            current_data = {}
        
        current_data[most_freq] = {
            "detection_score": csv_data['DetectionScore'],
            "ocr_confidence": csv_data['OcrConfidence'],
            "combined_score": csv_data['CombinedScore'],
            "img_path": image_path_relative,
            "status": db_plate.status
        }
        
        with open(full_json_path, 'w') as f:
            json.dump(current_data, f, indent=4)
        
        print(f"💾 Saved detection: {most_freq} ({db_plate.status}) -> {date_str}/")
        processed_detections[track_id] = True

def anpr_worker(rtsp_link, output_json_path, coordinates_string, camera_name, camera_id):
    """Ubuntu-optimized ANPR worker with robust RTSP handling"""
    atexit.register(cleanup_status_file, camera_id)
    
    frame_queue = Queue(maxsize=3)  # Small queue for low latency
    frame_reader = None
    frame_buffer = deque(maxlen=BUFFER_SIZE)
    processed_detections = {}
    
    # Variables to track actual frame properties
    actual_fps = VIDEO_FPS
    actual_width = 1920
    actual_height = 1080

    try:
        if ocr is None or model is None or mot_tracker is None:
            print("Error: Models or tracker not initialized.")
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
        
        buffer = 15
        
        print(f"🚀 Starting Ubuntu-optimized RTSP processing: {rtsp_link}")
        write_status(camera_id, "connecting")
        
        # Start threaded frame reader
        frame_reader = RTSPFrameReader(rtsp_link, frame_queue)
        frame_reader.start()
        
        # Wait for first frame and get actual dimensions
        print("Waiting for RTSP stream to stabilize...")
        wait_time = 0
        first_frame = None
        
        while wait_time < 30:
            try:
                first_frame = frame_queue.get(timeout=1.0)
                if first_frame is not None and first_frame.size > 0:
                    actual_height, actual_width = first_frame.shape[:2]
                    print(f"📐 Detected frame dimensions: {actual_width}x{actual_height}")
                    frame_queue.put(first_frame)  # Put it back
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

        active_tracks = {}
        frame_count = 0
        last_stats = time.time()
        fps_start_time = time.time()
        
        while frame_reader.running:
            try:
                # Get frame from queue with timeout
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
                    print(f"📊 Processing stats - Frames: {frame_count}, FPS: {fps_display:.1f}, Actual FPS: {actual_fps:.1f}")
                    last_stats = time.time()
                    frame_count = 0
                
                ocr_frame = frame.copy()
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                
                # YOLO detection
                results = model(frame, verbose=False)
                detections_ = []
                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    for box, confidence in zip(boxes, confidences):
                        x1, y1, x2, y2 = box
                        detections_.append([x1, y1, x2, y2, confidence])
                
                # SORT tracking
                track_ids = mot_tracker.update(np.asarray(detections_)) if len(detections_) > 0 else np.empty((0, 5))
                
                # Initialize in_track_ids for this frame
                in_track_ids = []
                
                if len(track_ids) > 0:
                    for track in track_ids:
                        x1, y1, x2, y2, track_id = track
                        track_id = int(track_id)
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        if x2 >= x_min and x1 <= x_max and y2 >= y_min and y1 <= y_max:
                            in_track_ids.append(track_id)
                            confidence = 0.85
                            
                            # Crop detection with bounds checking
                            y1_crop = max(0, y1-buffer)
                            y2_crop = min(ocr_frame.shape[0], y2+buffer)
                            x1_crop = max(0, x1-buffer)
                            x2_crop = min(ocr_frame.shape[1], x2+buffer)
                            
                            cropped_image = ocr_frame[y1_crop:y2_crop, x1_crop:x2_crop]
                            
                            # Skip if crop is too small
                            if cropped_image.shape[0] < 20 or cropped_image.shape[1] < 20:
                                continue
                            
                            # OCR processing
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
                                print(f"OCR processing error: {e}")
                            
                            if track_id not in active_tracks:
                                active_tracks[track_id] = []
                            
                            active_tracks[track_id].append({
                                'text': ocr_res.replace(" ", ""),
                                'confidence': ocr_confidence,
                                'detection_score': confidence,
                                'combined_score': (ocr_confidence + confidence) / 2,
                                'cropped_image': cropped_image.copy()
                            })

                # Process tracks that have left the scene
                leaving_ids = list(set(active_tracks.keys()) - set(in_track_ids))
                for leaving_id in leaving_ids:
                    if leaving_id not in processed_detections:
                        process_and_save_detection(
                            leaving_id, 
                            active_tracks, 
                            camera_name, 
                            actual_fps,  # Use calculated actual FPS
                            actual_width,  # Use detected frame width
                            actual_height,  # Use detected frame height
                            list(frame_buffer)
                        )
                    del active_tracks[leaving_id]

            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

    except Exception as e:
        write_status(camera_id, "error")
        print(f"❌ Error in anpr_worker for {rtsp_link}: {e}")
    finally:
        # Cleanup
        if frame_reader:
            frame_reader.stop()
        print(f"🏁 Stopped processing: {rtsp_link}")
