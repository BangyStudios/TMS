import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch
import easyocr
import re

class CarDetector:
    def __init__(self, save_dir='detected_cars', model_type='yolov8m-seg.pt'):
        # Set up device
        if torch.backends.mps.is_available():
            self.device = 'mps'
            print("Using MPS (Apple Metal) device")
        elif torch.cuda.is_available():
            self.device = 'cuda'
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            print("Using CPU (no GPU acceleration available)")
            
        # Initialize YOLO model
        self.model = YOLO(model_type)
        self.model.to(self.device)
        
        # Create directory for saving detected cars
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Class ID for cars in COCO dataset
        self.car_class_id = 2
        
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'], gpu=torch.mps.is_available())
        
        self.element_structure = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(15, 5))
        
    def detect_license_plate(self, car_crop):
        # Convert the car crop to grayscale
        gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
        
        # Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use Canny edge detection to find edges
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours in the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.imshow("Blurred", blurred)
        cv2.imshow("Edges", edges)
        cv2.waitKey(0)
        
        # Iterate through contours to find potential license plates
        for contour in contours:
            # Approximate the contour to a polygon
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            
            # Check if the polygon has 4 vertices (likely a rectangle)
            if len(approx) == 4:
                # Get the bounding box of the contour
                x, y, w, h = cv2.boundingRect(approx)
                
                # Check if the aspect ratio is close to a license plate
                aspect_ratio = float(w) / h
                if 2.5 < aspect_ratio < 5.5:
                    # Crop the license plate region
                    license_plate = car_crop[y:y+h, x:x+w]
                    
                    # Display the license plate
                    cv2.imshow("License Plate", license_plate)
                    cv2.waitKey(0)
                    return license_plate
        return None
        
    def process_frame(self, frame, frame_count=None):
        if frame is None:
            return []
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
            
        # Run YOLO detection with segmentation
        results = self.model(frame, stream=True, device=self.device)
        
        car_images = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_suffix = f"_frame{frame_count}" if frame_count is not None else ""
        
        for r in results:
            boxes = r.boxes
            masks = r.masks
            
            if masks is None:
                continue
                
            # Process each detection
            for box, mask in zip(boxes, masks):
                # Check if detection is a car
                if int(box.cls) == self.car_class_id:
                    # Get confidence score
                    conf = float(box.conf)
                    if conf < 0.8:  # Confidence threshold
                        continue
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Crop the car region
                    car_crop = frame[y1:y2, x1:x2]
                    
                    # Detect license plate in the car crop
                    license_plate = self.detect_license_plate(car_crop)
                    
                    if license_plate is not None:
                        # Save the license plate image
                        license_filename = self.save_dir / f"license_plate_{timestamp}{frame_suffix}_{len(car_images)}.jpg"
                        cv2.imwrite(str(license_filename), license_plate)
                    
                    # Save the cropped car image
                    # car_filename = self.save_dir / f"car_{timestamp}{frame_suffix}_{len(car_images)}.jpg"
                    # cv2.imwrite(str(car_filename), car_crop)

    def process_video(self, source, display=True, skip_frames=0, motion_threshold=5000):
        # Handle both string paths and camera indices
        if isinstance(source, (str, Path)):
            source = str(Path(source))
            is_video_file = True
        else:
            is_video_file = False
            
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video_file else None
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create progress bar for video files
        pbar = tqdm(total=total_frames) if is_video_file else None
        
        frame_count = 0
        frame_skip_counter = 0
        previous_frame = None
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                frame_skip_counter += 1
                
                # Update progress bar
                if pbar is not None:
                    pbar.update(1)
                    
                # Skip frames if requested
                if skip_frames > 0 and frame_skip_counter % (skip_frames + 1) != 0:
                    continue
                    
                # Convert frame to grayscale for motion detection
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
                
                # Calculate motion if previous frame exists
                if previous_frame is not None:
                    frame_diff = cv2.absdiff(previous_frame, gray_frame)
                    _, threshold_diff = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                    motion_area = np.sum(threshold_diff > 0)
                    
                    if motion_area < motion_threshold:
                        # Skip frame if motion is below threshold
                        continue
                
                previous_frame = gray_frame
                
                # Process frame and get detected cars
                detected_cars = self.process_frame(frame, frame_count)
                
                # Display the frame with detections if requested
                if display:
                    # Resize large frames for display
                    max_display_dim = 1200
                    if frame_width > max_display_dim or frame_height > max_display_dim:
                        scale = max_display_dim / max(frame_width, frame_height)
                        display_frame = cv2.resize(frame, (int(frame_width * scale), int(frame_height * scale)))
                    else:
                        display_frame = frame
                        
                    cv2.imshow('Car Detection', display_frame)
                    
                    # Break loop on 'q' press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            if pbar is not None:
                pbar.close()
            if display:
                cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # Initialize detector
    detector = CarDetector()
    
    # Example usage with video file
    video_file = "20241109100005_20241109100241_13_main_Intrusion Detection.mp4"  # Replace with your video path
    
    # Process video file
    try:
        detector.process_video(
            source=video_file,  # Can be video file path or camera index (0)
            display=True,  # Set to False to disable video display
            skip_frames=6  # Process every 3rd frame (adjust as needed)
        )
    except ValueError as e:
        print(f"Error: {e}")