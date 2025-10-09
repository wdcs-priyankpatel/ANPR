# backend/app/detection_manager.py

import multiprocessing as mp
import os

from .anpr_worker import anpr_worker

# Global dictionary to store active detection processes
active_processes = {}

def start_detection(camera_id: int, rtsp_link: str, output_json_path: str, coordinates: str, camera_name: str):
    """Starts a new detection process for a camera."""
    if camera_id in active_processes and active_processes[camera_id].is_alive():
        return False, "Detection already running for this camera."

    print(f"Spawning new process for camera {camera_id}...")
    
    # Pass all necessary arguments, including the new camera_id
    process = mp.Process(target=anpr_worker, args=(rtsp_link, output_json_path, coordinates, camera_name, camera_id))
    process.start()
    active_processes[camera_id] = process
    return True, "Detection started successfully."

def stop_detection(camera_id: int):
    # ... (rest of the code is the same)
    if camera_id in active_processes:
        process = active_processes.pop(camera_id)
        if process.is_alive():
            process.terminate()
            process.join()
            return True, "Detection stopped successfully."
        return False, "Process was not running."
    return False, "No active process found for this camera."

def get_status(camera_id: int):
    # ... (rest of the code is the same)
    if camera_id in active_processes and active_processes[camera_id].is_alive():
        return "running"
    return "stopped"