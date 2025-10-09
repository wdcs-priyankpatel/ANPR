import cv2
import os
import json
import numpy as np
from ultralytics import YOLO
import time
from paddleocr import PaddleOCR
import re
import multiprocessing as mp
from multiprocessing import set_start_method
from sort.sort import *
import threading
from queue import Queue
import signal
import sys
import csv
import subprocess

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Initialize SORT tracker
mot_tracker = Sort()

DET_model_path = 'best_21k_v8.pt'
model = YOLO(DET_model_path, task="detect")

# Paths for saving results
path_to_frames = "./frames_01"
path_of_coordinates = './cordinates_1.txt'
path_to_json = "./results_json"
mid_frames = "./mid_frame_02"
output_video_path = "./output_video_live.avi"

device = "cpu"

# Create directories if they don't exist
for path in [path_to_frames, path_to_json, mid_frames]:
    if not os.path.exists(path):
        os.makedirs(path)

# Global variables for graceful shutdown
running = True
frame_queue = Queue(maxsize=3)  # Smaller queue for better performance

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\nShutting down gracefully...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

def read_coordinates(path_of_coordinates):
    """Read coordinates from file with error handling"""
    try:
        if not os.path.exists(path_of_coordinates):
            print(f"Warning: Coordinates file '{path_of_coordinates}' not found. Creating default coordinates file.")
            with open(path_of_coordinates, 'w') as file:
                file.write("100,100,300,200")
        
        with open(path_of_coordinates, 'r') as file:
            line = file.readline().strip()
            
            if not line:
                print(f"Warning: Coordinates file '{path_of_coordinates}' is empty. Using default coordinates.")
                with open(path_of_coordinates, 'w') as file:
                    file.write("100,100,300,200")
                line = "100,100,300,200"
            
            coords = line.split(',')
            if len(coords) != 4:
                raise ValueError(f"Expected 4 coordinates, got {len(coords)}")
            
            x_min, y_max, x_max, y_min = map(int, coords)
            
            if x_min >= x_max or y_min >= y_max:
                print("Warning: Invalid coordinates detected. x_min should be < x_max and y_min should be < y_max")
            
            return (x_min, y_max), (x_max, y_min)
            
    except Exception as e:
        print(f"Error reading coordinates: {e}")
        return (100, 100), (300, 200)

def test_camera_streams(base_rtsp_url):
    """Test different camera stream paths to find H.264 stream"""
    # Extract base URL without stream path
    if '/stream' in base_rtsp_url:
        base_url = base_rtsp_url.split('/stream')[0]
    else:
        base_url = base_rtsp_url.rsplit('/', 1)[0] if '/' in base_rtsp_url else base_rtsp_url
    
    # Common stream paths that usually provide H.264
    stream_paths = [
        '/h264',           # Common H.264 path
        '/stream2',        # Secondary stream (usually H.264)
        '/live/sub',       # Sub stream (usually H.264, lower quality)
        '/live/main',      # Main stream
        '/cam/realmonitor?channel=1&subtype=1',  # Dahua format
        '/profile2/media.smp',  # Some IP cameras
        '/onvif1',         # ONVIF stream
        '/video.h264',     # Direct H.264
        '/stream1',        # Keep original as fallback
    ]
    
    print("Testing different camera streams for H.264 compatibility...")
    
    for stream_path in stream_paths:
        test_url = base_url + stream_path
        print(f"\nTesting: {test_url}")
        
        try:
            cap = cv2.VideoCapture(test_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                print(f"❌ Could not connect to {stream_path}")
                cap.release()
                continue
            
            # Test frame reading for 3 seconds
            good_frames = 0
            hevc_errors = 0
            start_time = time.time()
            
            while time.time() - start_time < 3:  # Test for 3 seconds
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Quick corruption check
                    if len(frame.shape) == 3 and frame.shape[0] > 10 and frame.shape[1] > 10:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        variance = np.var(gray)
                        if variance > 50:  # Good variance indicates valid frame
                            good_frames += 1
                
                time.sleep(0.1)
            
            cap.release()
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Stream info - FPS: {fps}, Resolution: {width}x{height}")
            print(f"Good frames in 3 seconds: {good_frames}")
            
            if good_frames >= 15:  # At least 5 fps worth of good frames
                print(f"✅ {stream_path} - Good stream found!")
                return test_url
            elif good_frames > 0:
                print(f"⚠️ {stream_path} - Partial success ({good_frames} good frames)")
            else:
                print(f"❌ {stream_path} - No good frames")
                
        except Exception as e:
            print(f"❌ Error testing {stream_path}: {e}")
    
    print("\n❌ No suitable H.264 stream found")
    return None

def create_ffmpeg_rtsp_capture(rtsp_url):
    """Create RTSP capture using FFmpeg with H.264 force"""
    try:
        print("Attempting FFmpeg capture with H.264 codec preference...")
        
        # FFmpeg command to convert HEVC to H.264 on-the-fly
        ffmpeg_cmd = [
            'ffmpeg',
            '-rtsp_transport', 'tcp',  # Use TCP instead of UDP for reliability
            '-i', rtsp_url,
            '-c:v', 'libx264',  # Force H.264 encoding
            '-preset', 'ultrafast',  # Fast encoding
            '-tune', 'zerolatency',  # Low latency
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-an',  # No audio
            'pipe:1'
        ]
        
        # Start FFmpeg process
        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return process
        
    except Exception as e:
        print(f"Failed to create FFmpeg capture: {e}")
        return None

def frame_capture_thread_ffmpeg(rtsp_url, frame_queue):
    """Frame capture using FFmpeg subprocess to handle HEVC"""
    global running
    
    process = None
    reconnect_count = 0
    max_reconnects = 5
    
    while running and reconnect_count < max_reconnects:
        try:
            print(f"Starting FFmpeg capture (attempt {reconnect_count + 1})...")
            process = create_ffmpeg_rtsp_capture(rtsp_url)
            
            if process is None:
                print("Failed to start FFmpeg process")
                time.sleep(5)
                reconnect_count += 1
                continue
            
            # Determine frame size (assuming 1920x1080, adjust as needed)
            width, height = 1920, 1080
            frame_size = width * height * 3  # BGR = 3 bytes per pixel
            
            consecutive_failures = 0
            frames_received = 0
            
            print("FFmpeg capture started successfully")
            
            while running:
                try:
                    # Read frame data from FFmpeg
                    raw_frame = process.stdout.read(frame_size)
                    
                    if len(raw_frame) != frame_size:
                        consecutive_failures += 1
                        if consecutive_failures > 10:
                            print("Too many frame read failures, restarting FFmpeg...")
                            break
                        continue
                    
                    # Convert raw data to OpenCV frame
                    frame = np.frombuffer(raw_frame, dtype=np.uint8)
                    frame = frame.reshape((height, width, 3))
                    
                    consecutive_failures = 0
                    frames_received += 1
                    
                    # Add frame to queue
                    if not frame_queue.full():
                        frame_queue.put((frame.copy(), time.time()), block=False)
                    else:
                        # Remove old frame and add new one
                        try:
                            frame_queue.get_nowait()
                        except:
                            pass
                        frame_queue.put((frame.copy(), time.time()), block=False)
                    
                    if frames_received % 100 == 0:
                        print(f"FFmpeg: {frames_received} frames received")
                    
                    time.sleep(0.01)  # Small delay
                    
                except Exception as e:
                    consecutive_failures += 1
                    print(f"Frame processing error: {e}")
                    if consecutive_failures > 20:
                        break
            
        except Exception as e:
            print(f"FFmpeg capture error: {e}")
        
        finally:
            if process:
                process.terminate()
                process.wait()
                process = None
        
        reconnect_count += 1
        if running and reconnect_count < max_reconnects:
            print(f"Reconnecting FFmpeg in 3 seconds... (attempt {reconnect_count + 1})")
            time.sleep(3)
    
    print("FFmpeg capture thread stopped")

def create_optimized_rtsp_capture(rtsp_url):
    """Create optimized RTSP capture specifically for H.264"""
    try:
        print("Creating optimized OpenCV RTSP capture...")
        
        # Try with CAP_FFMPEG backend
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        # Optimized settings for H.264
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
        
        # Force codec settings
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS
        
        # Additional optimization
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Try lower resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if cap.isOpened():
            # Test frame reading
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print("✅ Optimized OpenCV capture successful")
                return cap
        
        cap.release()
        return None
        
    except Exception as e:
        print(f"Optimized capture failed: {e}")
        return None

def frame_capture_thread_optimized(rtsp_url, frame_queue):
    """Optimized frame capture thread"""
    global running
    
    cap = None
    reconnect_attempts = 0
    max_reconnects = 3
    
    while running and reconnect_attempts < max_reconnects:
        try:
            # First, try to find a good stream
            if reconnect_attempts == 0:
                print("Searching for optimal camera stream...")
                optimal_url = test_camera_streams(rtsp_url)
                if optimal_url:
                    rtsp_url = optimal_url
                    print(f"Using optimal stream: {rtsp_url}")
                else:
                    print("No optimal stream found, using original URL")
            
            cap = create_optimized_rtsp_capture(rtsp_url)
            
            if cap is None:
                print("OpenCV capture failed, falling back to FFmpeg...")
                frame_capture_thread_ffmpeg(rtsp_url, frame_queue)
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Capture started - FPS: {fps}, Resolution: {width}x{height}")
            
            consecutive_failures = 0
            frames_captured = 0
            last_stats_time = time.time()
            
            while running:
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        print("Too many read failures, reconnecting...")
                        break
                    time.sleep(0.05)
                    continue
                
                # Basic frame validation
                if frame.shape[0] < 100 or frame.shape[1] < 100:
                    consecutive_failures += 1
                    continue
                
                consecutive_failures = 0
                frames_captured += 1
                
                # Add frame to queue
                try:
                    if not frame_queue.full():
                        frame_queue.put((frame.copy(), time.time()), block=False)
                    else:
                        # Replace oldest frame
                        try:
                            frame_queue.get_nowait()
                        except:
                            pass
                        frame_queue.put((frame.copy(), time.time()), block=False)
                except:
                    pass
                
                # Print stats every 5 seconds
                if time.time() - last_stats_time > 5:
                    actual_fps = frames_captured / (time.time() - last_stats_time + 5)
                    print(f"Capture stats - Frames: {frames_captured}, FPS: {actual_fps:.1f}")
                    last_stats_time = time.time()
                    frames_captured = 0
                
                time.sleep(0.02)  # ~50 FPS max
        
        except Exception as e:
            print(f"Capture error: {e}")
        
        finally:
            if cap:
                cap.release()
                cap = None
        
        reconnect_attempts += 1
        if running and reconnect_attempts < max_reconnects:
            print(f"Reconnecting in 5 seconds... (attempt {reconnect_attempts + 1})")
            time.sleep(5)
    
    print("Frame capture thread ended")

# Keep all your existing functions for ANPR processing
def get_best_detection_for_track(ocr_results_list):
    """Get the detection with highest combined accuracy score"""
    if not ocr_results_list:
        return None
    
    best_idx = max(range(len(ocr_results_list)), 
                   key=lambda i: ocr_results_list[i]['combined_score'])
    return ocr_results_list[best_idx]

def get_most_frequent_text_from_best_detections(ocr_results_list):
    """Get most frequent text from best detections based on accuracy"""
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

def anpr_rtsp_live(rtsp_url, output_csv_path, save_video=True, display_video=True):
    """Main ANPR function with improved RTSP handling"""
    global running
    
    try:
        # Initialize CSV file
        csv_file_path = os.path.join(path_to_json, output_csv_path)
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['plate_number', 'detection_score', 'ocr_confidence', 'combined_score', 'timestamp', 'img_path_0'])

        rect_start, rect_end = read_coordinates(path_of_coordinates)
        
        x_min = min(rect_start[0], rect_end[0])
        y_min = min(rect_start[1], rect_end[1])
        x_max = max(rect_start[0], rect_end[0])
        y_max = max(rect_start[1], rect_end[1])
        
        buffer = 15
        detection_scores_per_id = {}
        ocr_results_per_id = {}
        results_dict = {}
        in_track_ids = []
        prev_track_ids = []
        prev_missing_ids = {}
        frames_track_ids = {}

        # Start optimized frame capture
        capture_thread = threading.Thread(target=frame_capture_thread_optimized, args=(rtsp_url, frame_queue))
        capture_thread.daemon = True
        capture_thread.start()
        
        # Wait for stream to start
        print("Waiting for camera stream...")
        wait_start = time.time()
        while frame_queue.empty() and running and (time.time() - wait_start < 30):
            time.sleep(0.5)
        
        if frame_queue.empty():
            print("❌ Timeout waiting for camera stream")
            return
        
        # Get first frame for video setup
        first_frame, _ = frame_queue.get()
        frame_height, frame_width = first_frame.shape[:2]
        print(f"✅ Stream started - Frame size: {frame_width}x{frame_height}")
        
        # Initialize video writer
        out = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_video_path, fourcc, 10.0, (frame_width, frame_height))
        
        frame_queue.put((first_frame, time.time()))  # Put frame back
        
        print("🚀 Starting ANPR processing...")
        print("Press 'q' to quit")
        
        frame_count = 0
        last_save_time = time.time()
        processing_times = []
        
        while running:
            try:
                if frame_queue.empty():
                    time.sleep(0.01)
                    continue
                
                frame_start_time = time.time()
                frame, frame_timestamp = frame_queue.get(timeout=1)
                frame_count += 1
                
                ocr_frame = frame.copy()
                to_save = False
                
                # Draw ROI
                cv2.rectangle(frame, rect_start, rect_end, (255, 0, 0), 2)
                
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
                if len(detections_) > 0:
                    track_ids = mot_tracker.update(np.asarray(detections_))
                else:
                    track_ids = np.empty((0, 5))
                
                # Process detections (your existing logic)
                if len(track_ids) > 0:
                    in_track_ids = []
                    
                    for track in track_ids:
                        x1, y1, x2, y2, track_id = track
                        track_id = int(track_id)
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        if x2 >= x_min and x1 <= x_max and y2 >= y_min and y1 <= y_max:
                            in_track_ids.append(track_id)
                            
                            confidence = 0.85
                            
                            if track_id not in detection_scores_per_id:
                                detection_scores_per_id[track_id] = []
                            detection_scores_per_id[track_id].append(confidence)
                            
                            # Crop detection
                            x1_buffered = max(0, x1 - buffer)
                            y1_buffered = max(0, y1 - buffer)
                            x2_buffered = min(frame.shape[1], x2 + buffer)
                            y2_buffered = min(frame.shape[0], y2 + buffer)
                            cropped_image = ocr_frame[y1_buffered:y2_buffered, x1_buffered:x2_buffered]
                            
                            if cropped_image.shape[0] < 20 or cropped_image.shape[1] < 20:
                                continue
                            
                            # OCR processing
                            temp_image_path = f"temp_crop_{track_id}_{time.time()}.jpg"
                            cv2.imwrite(temp_image_path, cropped_image)
                            
                            ocr_result = ocr.ocr(temp_image_path, cls=True)
                            ocr_res = ""
                            ocr_confidence = 0.0
                            
                            try:
                                if ocr_result:
                                    for line in ocr_result:
                                        if line:
                                            for word_info in line:
                                                ocr_res += word_info[1][0] + " "
                                                ocr_confidence += word_info[1][1]
                                    if len([word for line in ocr_result if line for word in line]) > 0:
                                        ocr_confidence /= len([word for line in ocr_result if line for word in line])
                            except Exception as e:
                                print(f"OCR error for track {track_id}: {e}")
                            
                            try:
                                os.remove(temp_image_path)
                            except:
                                pass
                            
                            if track_id not in ocr_results_per_id:
                                ocr_results_per_id[track_id] = []
                            
                            ocr_results_per_id[track_id].append({
                                'text': ocr_res.replace(" ", ""),
                                'confidence': ocr_confidence,
                                'detection_score': confidence,
                                'combined_score': (ocr_confidence + confidence) / 2,
                                'cropped_image': cropped_image.copy(),
                                'crop_path': None
                            })
                            
                            # Draw detection
                            cv2.rectangle(frame, (x1_buffered, y1_buffered), (x2_buffered, y2_buffered), (0, 255, 0), 3)
                            cv2.putText(frame, f"ID: {track_id}", (x1_buffered, y1_buffered - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            
                            # Save best crop
                            if len(ocr_results_per_id[track_id]) == 1:
                                crop_name = f"crop_{track_id}_{time.time()}.jpg"
                                crop_path = os.path.join(path_to_frames, crop_name)
                                cv2.imwrite(crop_path, cropped_image)
                                ocr_results_per_id[track_id][-1]['crop_path'] = crop_path
                
                # Handle missing tracks (your existing logic)
                missing_ids = missing_elements(prev_track_ids, in_track_ids)
                
                if len(missing_ids) > 0:
                    for missing_id in missing_ids:
                        if missing_id not in prev_missing_ids:
                            prev_missing_ids[missing_id] = 0
                
                for prev_missing_id in list(prev_missing_ids.keys()):
                    if prev_missing_id not in in_track_ids:
                        prev_missing_ids[prev_missing_id] += 1
                        if prev_missing_ids[prev_missing_id] >= 30:
                            if prev_missing_id in ocr_results_per_id:
                                most_freq = get_most_frequent_text_from_best_detections(ocr_results_per_id[prev_missing_id])
                                best_detection = get_best_detection_for_track(ocr_results_per_id[prev_missing_id])
                                
                                if most_freq != " " and most_freq != "" and most_freq not in results_dict and best_detection:
                                    crop_path = best_detection['crop_path']
                                    img_path_dict = {"img_path_0": crop_path} if crop_path else {}
                                    
                                    results_dict[most_freq] = {
                                        "detection_score": round(best_detection['detection_score'], 4),
                                        "ocr_confidence": round(best_detection['confidence'], 4),
                                        "combined_score": round(best_detection['combined_score'], 4),
                                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        **img_path_dict
                                    }
                                    
                                    print(f"🎯 Number plate detected: {most_freq}")
                            
                            to_save = True
                            prev_missing_ids.pop(prev_missing_id)
                    else:
                        prev_missing_ids.pop(prev_missing_id)
                
                # Save results periodically
                current_time = time.time()
                if to_save or (current_time - last_save_time) > 5:
                    with open(csv_file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['plate_number', 'detection_score', 'ocr_confidence', 'combined_score', 'timestamp', 'img_path_0'])
                        for plate, data in results_dict.items():
                            writer.writerow([plate, data['detection_score'], data['ocr_confidence'], 
                                          data['combined_score'], data['timestamp'], data.get('img_path_0', '')])
                    last_save_time = current_time
                
                prev_track_ids = in_track_ids
                
                # Add frame info
                processing_time = time.time() - frame_start_time
                processing_times.append(processing_time)
                if len(processing_times) > 30:
                    processing_times.pop(0)
                
                avg_processing_time = sum(processing_times) / len(processing_times)
                fps_estimate = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
                
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, f"LIVE - {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"FPS: {fps_estimate:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Plates: {len(results_dict)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Save video frame
                if save_video and out is not None:
                    out.write(frame)
                
                # Display frame
                if display_video:
                    cv2.imshow('RTSP ANPR Live Feed - H.264 Optimized', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quit key pressed")
                        running = False
                        break
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                continue
    
    except Exception as e:
        print(f"Error in ANPR processing: {e}")
    
    finally:
        running = False
        
        # Process remaining tracks
        print("Processing remaining tracks...")
        for track_id in ocr_results_per_id:
            if track_id not in results_dict:
                most_freq = get_most_frequent_text_from_best_detections(ocr_results_per_id[track_id])
                best_detection = get_best_detection_for_track(ocr_results_per_id[track_id])
                
                if most_freq != " " and most_freq != "" and best_detection:
                    crop_path = best_detection['crop_path']
                    img_path_dict = {"img_path_0": crop_path} if crop_path else {}
                    
                    results_dict[most_freq] = {
                        "detection_score": round(best_detection['detection_score'], 4),
                        "ocr_confidence": round(best_detection['confidence'], 4),
                        "combined_score": round(best_detection['combined_score'], 4),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        **img_path_dict
                    }
        
        # Final save
        if results_dict:
            with open(csv_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['plate_number', 'detection_score', 'ocr_confidence', 'combined_score', 'timestamp', 'img_path_0'])
                for plate, data in results_dict.items():
                    writer.writerow([plate, data['detection_score'], data['ocr_confidence'], 
                                  data['combined_score'], data['timestamp'], data.get('img_path_0', '')])
            
            print(f"✅ Final results saved: {csv_file_path}")
            print(f"📊 Total plates detected: {len(results_dict)}")
            for plate in results_dict.keys():
                print(f"   - {plate}")
        
        # Cleanup
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        print("🏁 ANPR processing completed")

def check_ffmpeg_availability():
    """Check if FFmpeg is available on the system"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg is available")
            return True
        else:
            print("❌ FFmpeg not working properly")
            return False
    except Exception as e:
        print(f"❌ FFmpeg not found: {e}")
        print("Install FFmpeg: https://ffmpeg.org/download.html")
        return False

def test_rtsp_comprehensive(rtsp_url):
    """Comprehensive RTSP testing"""
    print("🔍 Comprehensive RTSP Stream Analysis")
    print("=" * 50)
    
    # Test basic connectivity
    print("1. Testing basic connectivity...")
    try:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        
        if cap.isOpened():
            print("✅ Basic connection successful")
            
            # Get stream info
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            
            # Convert fourcc to readable format
            fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            print(f"   📹 Stream Info:")
            print(f"      - Resolution: {width}x{height}")
            print(f"      - FPS: {fps}")
            print(f"      - Codec: {fourcc_str}")
            
            # Test frame reading
            print("2. Testing frame reading...")
            good_frames = 0
            bad_frames = 0
            hevc_detected = False
            
            for i in range(20):  # Test 20 frames
                ret, frame = cap.read()
                if ret and frame is not None:
                    if frame.shape[0] > 100 and frame.shape[1] > 100:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        variance = np.var(gray)
                        if variance > 100:  # Good frame
                            good_frames += 1
                        else:
                            bad_frames += 1
                    else:
                        bad_frames += 1
                else:
                    bad_frames += 1
                
                time.sleep(0.1)
            
            print(f"   📊 Frame Quality: {good_frames}/20 good, {bad_frames}/20 bad")
            
            cap.release()
            
            if good_frames >= 15:
                print("✅ Stream quality is GOOD")
                return True
            elif good_frames >= 8:
                print("⚠️ Stream quality is MODERATE - may have issues")
                return True
            else:
                print("❌ Stream quality is POOR")
                return False
        else:
            print("❌ Failed to connect")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 H.264 Optimized RTSP ANPR System")
    print("=" * 50)
    
    # Check system requirements
    print("Checking system requirements...")
    ffmpeg_available = check_ffmpeg_availability()
    
    # RTSP URL
    rtsp_url = "rtsp://admin:admin@1234@192.168.11.106:554/stream1"
    
    if len(sys.argv) > 1:
        rtsp_url = sys.argv[1]
    
    print(f"📺 Target RTSP URL: {rtsp_url}")
    
    # Comprehensive testing
    if not test_rtsp_comprehensive(rtsp_url):
        print("\n💡 Troubleshooting suggestions:")
        print("1. Try different stream paths:")
        print("   - rtsp://admin:admin@1234@192.168.11.106:554/h264")
        print("   - rtsp://admin:admin@1234@192.168.11.106:554/stream2")
        print("   - rtsp://admin:admin@1234@192.168.11.106:554/live/sub")
        
        print("2. Change camera settings:")
        print("   - Set video codec to H.264 (not H.265/HEVC)")
        print("   - Reduce bitrate to 2-4 Mbps")
        print("   - Lower resolution to 1280x720 or 1920x1080")
        
        print("3. Check network:")
        print("   - Ensure stable network connection")
        print("   - Try wired connection instead of WiFi")
        
        if not ffmpeg_available:
            print("4. Install FFmpeg for better codec support")
        
        # Still proceed with testing but with warnings
        proceed = input("\nProceed anyway? (y/N): ")
        if proceed.lower() != 'y':
            sys.exit(1)
    
    # Start ANPR processing
    try:
        print("\n🎬 Starting ANPR processing...")
        anpr_rtsp_live(
            rtsp_url=rtsp_url,
            output_csv_path="live_anpr_results.csv",
            save_video=True,
            display_video=True
        )
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("👋 Program ended")