"""
Posture Analyzer Module - MediaPipe Tasks API (Fixed Import)
Uses the new MediaPipe Vision Tasks API for pose detection
"""

import cv2
import numpy as np
import math
import time
from typing import Tuple, Optional, Dict, List
import os

# Import MediaPipe Tasks API with correct imports
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core import base_options as base_options_module
    MEDIAPIPE_AVAILABLE = True
    print("✓ MediaPipe Tasks API imported successfully")
except ImportError as e:
    print(f"⚠️ MediaPipe Tasks API import error: {e}")
    print("   Install: pip install mediapipe-tasks")
    MEDIAPIPE_AVAILABLE = False
except Exception as e:
    print(f"⚠️ MediaPipe import error: {e}")
    MEDIAPIPE_AVAILABLE = False

class Analyzer:
    """
    Analyzes sitting posture using MediaPipe Tasks API.
    Modern, efficient, and works with OpenCV 5.0.0+
    """
    
    def __init__(self, threshold: float = 15.0, 
                 model_path: str = "models/pose_landmarker_lite.task"):
        """
        Initialize the posture analyzer with MediaPipe Tasks API.
        
        Args:
            threshold (float): Angle threshold for poor posture detection in degrees.
            model_path (str): Path to the pose landmarker model file.
        """
        self.threshold = threshold
        self.total_frames = 0
        self.good_posture_frames = 0
        self.bad_posture_frames = 0
        self.posture_history = []
        self.max_history = 100
        self.processing_times = []
        self.fps = 0
        self.detection_success_count = 0
        self.detection_fail_count = 0
        
        # MediaPipe Task API components
        self.detector = None
        self.use_mediapipe = False
        
        # Landmark indices for MediaPipe (33 landmarks)
        self.LANDMARK_INDICES = {
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_ear': 7,
            'right_ear': 8,
            'left_hip': 23,
            'right_hip': 24,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'nose': 0,
            'left_eye': 2,
            'right_eye': 5,
        }
        
        # Check if model exists, download if not
        if not os.path.exists(model_path):
            print(f"\n⚠️ Model file not found: {model_path}")
            print("   Attempting to download default model...")
            downloaded = self._download_default_model()
            if downloaded:
                model_path = downloaded
            else:
                print("   Please download the model manually:")
                print("   https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
                print("   Or run: python download_pose_model.py")
        
        # Initialize MediaPipe if available
        if MEDIAPIPE_AVAILABLE and os.path.exists(model_path):
            self._init_mediapipe_tasks(model_path)
        else:
            print("\n⚠️ MediaPipe Tasks API not available or model not found.")
            print("   Using simplified detection.")
            if not MEDIAPIPE_AVAILABLE:
                print("   Install: pip install mediapipe-tasks")
    
    def _download_default_model(self):
        """Download the default pose landmarker model"""
        try:
            import requests
            import os
            
            # Create models directory
            os.makedirs('models', exist_ok=True)
            
            model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            model_path = "models/pose_landmarker_lite.task"
            
            print(f"   Downloading from: {model_url}")
            
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r   Progress: {progress:.1f}%", end='')
            
            print(f"\n   ✓ Model downloaded successfully!")
            return model_path
            
        except ImportError:
            print("   ✗ requests module not found. Please install: pip install requests")
            return None
        except Exception as e:
            print(f"   ✗ Download error: {e}")
            return None
    
    def _init_mediapipe_tasks(self, model_path: str):
        """
        Initialize the MediaPipe PoseLandmarker using the Tasks API.
        
        Args:
            model_path (str): Path to the model file.
        """
        try:
            # Create base options with model file
            base_options = python.BaseOptions(
                model_asset_path=model_path
            )
            
            # Create PoseLandmarker options
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False
            )
            
            # Create the detector
            self.detector = vision.PoseLandmarker.create_from_options(options)
            self.use_mediapipe = True
            print(f"✓ MediaPipe Tasks API PoseLandmarker initialized successfully")
            print(f"   Model: {model_path}")
            
        except Exception as e:
            print(f"⚠️ MediaPipe Tasks API initialization error: {e}")
            print("   Falling back to simplified detection.")
            self.use_mediapipe = False
    
    def _convert_frame_to_mp_image(self, frame: np.ndarray):
        """
        Convert OpenCV frame to MediaPipe Image using the correct API.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            MediaPipe Image object
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image using the correct method
        # Using the Image class from mediapipe
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return mp_image
    
    def analyze_posture(self, frame: np.ndarray) -> Tuple[bool, float, Optional[Dict]]:
        """
        Analyze posture from a single frame.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Tuple: (is_good_posture, angle, landmarks)
        """
        start_time = time.time()
        
        if self.use_mediapipe and self.detector is not None:
            is_good, angle, landmarks = self._analyze_with_tasks_api(frame)
        else:
            is_good, angle, landmarks = self._analyze_simplified(frame)
        
        # Update statistics
        self.total_frames += 1
        if is_good:
            self.good_posture_frames += 1
        else:
            self.bad_posture_frames += 1
            
        self.posture_history.append({
            'timestamp': time.time(),
            'angle': angle,
            'is_good': is_good
        })
        if len(self.posture_history) > self.max_history:
            self.posture_history.pop(0)
        
        # Calculate FPS
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)
            self.fps = 1.0 / (sum(self.processing_times) / len(self.processing_times))
        
        return is_good, angle, landmarks
    
    def _analyze_with_tasks_api(self, frame: np.ndarray) -> Tuple[bool, float, Optional[Dict]]:
        """
        Analyze posture using the MediaPipe Tasks API.
        """
        try:
            # Convert frame to MediaPipe Image
            mp_image = self._convert_frame_to_mp_image(frame)
            
            # Detect pose
            detection_result = self.detector.detect(mp_image)
            
            is_good = True
            angle = 0.0
            landmarks = None
            
            # Check if pose detected
            if detection_result.pose_landmarks:
                # Get first pose
                pose_landmarks = detection_result.pose_landmarks[0]
                
                # Extract landmarks
                landmarks = self._extract_landmarks(pose_landmarks, frame.shape)
                
                if landmarks.get('left_shoulder') and landmarks.get('right_shoulder'):
                    angle = self._calculate_shoulder_angle(landmarks)
                    is_good = angle <= self.threshold
                    self.detection_success_count += 1
                else:
                    self.detection_fail_count += 1
            else:
                self.detection_fail_count += 1
            
            return is_good, angle, landmarks
            
        except Exception as e:
            print(f"⚠️ MediaPipe Tasks API error: {e}")
            self.detection_fail_count += 1
            return True, 0.0, None
    
    def _analyze_simplified(self, frame: np.ndarray) -> Tuple[bool, float, Optional[Dict]]:
        """
        Simplified posture analysis as fallback.
        """
        try:
            # Use face detection as a proxy
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                self.detection_success_count += 1
                return True, 0.0, None
            else:
                self.detection_fail_count += 1
                return False, 30.0, None
                
        except Exception as e:
            self.detection_fail_count += 1
            return True, 0.0, None
    
    def _extract_landmarks(self, pose_landmarks, frame_shape) -> Dict[str, Tuple[float, float]]:
        """
        Extract relevant landmarks from MediaPipe Tasks API results.
        
        Args:
            pose_landmarks: Pose landmark list from MediaPipe
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            Dictionary of landmark positions in pixel coordinates
        """
        landmarks = {}
        h, w = frame_shape[:2]
        
        for name, idx in self.LANDMARK_INDICES.items():
            if idx < len(pose_landmarks):
                lm = pose_landmarks[idx]
                # Convert normalized coordinates to pixel coordinates
                landmarks[name] = (lm.x * w, lm.y * h)
        
        return landmarks
    
    def _calculate_shoulder_angle(self, landmarks: Dict) -> float:
        """
        Calculate shoulder angle from landmarks.
        
        Returns:
            Angle in degrees from horizontal (0° = horizontal, 90° = vertical)
        """
        try:
            left_shoulder = landmarks.get('left_shoulder')
            right_shoulder = landmarks.get('right_shoulder')
            
            if left_shoulder is None or right_shoulder is None:
                return 0.0
            
            lx, ly = left_shoulder
            rx, ry = right_shoulder
            
            dx = rx - lx
            dy = ry - ly
            
            if dx == 0:
                return 90.0
            
            angle_rad = math.atan2(abs(dy), abs(dx))
            angle_deg = math.degrees(angle_rad)
            
            return angle_deg
            
        except Exception as e:
            print(f"Error calculating angle: {e}")
            return 0.0
    
    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw pose landmarks on the frame using the Tasks API.
        """
        if not self.use_mediapipe or self.detector is None:
            return frame
        
        try:
            # Convert frame to MediaPipe Image
            mp_image = self._convert_frame_to_mp_image(frame)
            
            # Detect pose
            detection_result = self.detector.detect(mp_image)
            
            # Draw landmarks if detected
            if detection_result.pose_landmarks:
                h, w = frame.shape[:2]
                
                # Draw landmarks as circles
                for idx, lm in enumerate(detection_result.pose_landmarks[0]):
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    
                    # Color based on body part
                    if idx in [11, 12]:  # Shoulders
                        color = (0, 255, 255)  # Yellow
                        radius = 6
                    elif idx in [13, 14, 15, 16]:  # Arms
                        color = (255, 0, 255)  # Magenta
                        radius = 4
                    elif idx in [23, 24]:  # Hips
                        color = (255, 255, 0)  # Cyan
                        radius = 5
                    elif idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # Face
                        color = (0, 0, 255)  # Red
                        radius = 3
                    else:
                        color = (0, 255, 0)  # Green
                        radius = 3
                    
                    cv2.circle(frame, (x, y), radius, color, -1)
                
                # Draw connections
                for connection in self._get_pose_connections():
                    idx1, idx2 = connection
                    if (idx1 < len(detection_result.pose_landmarks[0]) and 
                        idx2 < len(detection_result.pose_landmarks[0])):
                        lm1 = detection_result.pose_landmarks[0][idx1]
                        lm2 = detection_result.pose_landmarks[0][idx2]
                        x1, y1 = int(lm1.x * w), int(lm1.y * h)
                        x2, y2 = int(lm2.x * w), int(lm2.y * h)
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
        except Exception as e:
            pass
        
        return frame
    
    def _get_pose_connections(self) -> List[Tuple[int, int]]:
        """
        Get the standard pose connections.
        MediaPipe Pose has 33 landmarks with specific connections.
        """
        return [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            # Body
            (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (27, 29), (29, 31), (24, 26), (26, 28), (28, 30), (30, 32),
            # Arms
            (11, 21), (12, 22), (21, 19), (22, 20), (19, 17), (20, 18),
        ]
    
    def draw_posture_info(self, frame: np.ndarray, is_good: bool, angle: float, 
                          landmarks: Optional[Dict] = None) -> np.ndarray:
        """
        Draw posture information on the frame.
        """
        h, w = frame.shape[:2]
        
        # Status box
        cv2.rectangle(frame, (10, 10), (300, 170), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 170), (255, 255, 255), 2)
        
        status = "✓ GOOD POSTURE" if is_good else "✗ BAD POSTURE"
        color = (0, 255, 0) if is_good else (0, 0, 255)
        cv2.putText(frame, status, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Angle: {angle:.1f}°", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {self.threshold}°", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        method = "MediaPipe Tasks API" if self.use_mediapipe else "Simplified"
        cv2.putText(frame, f"Method: {method}", (20, 155), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Statistics box
        score = self.calculate_posture_score()
        cv2.rectangle(frame, (w - 300, 10), (w - 10, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (w - 300, 10), (w - 10, 150), (255, 255, 255), 2)
        cv2.putText(frame, f"Score: {score:.1f}%", (w - 290, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Progress bar
        bar_width = 260
        bar_height = 20
        bar_x = w - 290
        bar_y = 55
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        fill_width = int((score / 100) * bar_width)
        color = (0, 255, 0) if score > 70 else (0, 255, 255) if score > 40 else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        total = self.total_frames
        good = self.good_posture_frames
        cv2.putText(frame, f"Good: {good}  Bad: {total - good}  Total: {total}", 
                   (w - 290, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw shoulder line if landmarks available
        if landmarks and 'left_shoulder' in landmarks and 'right_shoulder' in landmarks:
            lx, ly = int(landmarks['left_shoulder'][0]), int(landmarks['left_shoulder'][1])
            rx, ry = int(landmarks['right_shoulder'][0]), int(landmarks['right_shoulder'][1])
            
            cv2.line(frame, (lx, ly), (rx, ry), (0, 255, 255), 2)
            
            center_x = (lx + rx) // 2
            center_y = (ly + ry) // 2
            cv2.ellipse(frame, (center_x, center_y), (50, 50), 0, 0, angle, 
                       (0, 255, 255), 2)
        
        return frame
    
    def draw_angle_visualization(self, frame: np.ndarray, angle: float) -> np.ndarray:
        """
        Draw a gauge showing current angle.
        """
        h, w = frame.shape[:2]
        gauge_x = 50
        gauge_y = h - 80
        gauge_radius = 40
        
        cv2.circle(frame, (gauge_x, gauge_y), gauge_radius, (50, 50, 50), -1)
        cv2.circle(frame, (gauge_x, gauge_y), gauge_radius, (255, 255, 255), 2)
        
        angle_rad = math.radians(angle - 90)
        end_x = int(gauge_x + gauge_radius * 0.8 * math.cos(angle_rad))
        end_y = int(gauge_y + gauge_radius * 0.8 * math.sin(angle_rad))
        
        if angle <= self.threshold:
            color = (0, 255, 0)
        elif angle <= self.threshold * 1.5:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        
        cv2.line(frame, (gauge_x, gauge_y), (end_x, end_y), color, 3)
        cv2.putText(frame, f"{angle:.1f}°", (gauge_x - 25, gauge_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def calculate_posture_score(self) -> float:
        """Calculate overall posture score based on recent history."""
        if self.total_frames == 0:
            return 100.0
        
        good_percentage = (self.good_posture_frames / self.total_frames) * 100
        
        recent_good = 0
        if self.posture_history:
            recent_frames = self.posture_history[-20:]
            recent_good = sum(1 for p in recent_frames if p['is_good'])
            recent_percentage = (recent_good / len(recent_frames)) * 100 if recent_frames else 0
            score = (good_percentage * 0.3) + (recent_percentage * 0.7)
        else:
            score = good_percentage
        
        return min(100.0, max(0.0, score))
    
    def get_statistics(self) -> Dict:
        """Get posture statistics."""
        if self.total_frames == 0:
            return {
                'total_frames': 0,
                'good_frames': 0,
                'bad_frames': 0,
                'good_percentage': 0,
                'bad_percentage': 0,
                'score': 100.0,
                'average_angle': 0.0,
                'detection_success_rate': 0.0,
                'method': 'mediapipe_tasks' if self.use_mediapipe else 'simplified'
            }
        
        avg_angle = 0
        if self.posture_history:
            avg_angle = sum(p['angle'] for p in self.posture_history) / len(self.posture_history)
        
        good_pct = (self.good_posture_frames / self.total_frames) * 100
        total_detections = self.detection_success_count + self.detection_fail_count
        success_rate = (self.detection_success_count / total_detections) * 100 if total_detections > 0 else 0
        
        return {
            'total_frames': self.total_frames,
            'good_frames': self.good_posture_frames,
            'bad_frames': self.bad_posture_frames,
            'good_percentage': good_pct,
            'bad_percentage': 100 - good_pct,
            'score': self.calculate_posture_score(),
            'average_angle': avg_angle,
            'fps': self.fps,
            'detection_success_rate': success_rate,
            'method': 'mediapipe_tasks' if self.use_mediapipe else 'simplified'
        }
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.total_frames = 0
        self.good_posture_frames = 0
        self.bad_posture_frames = 0
        self.posture_history = []
        self.processing_times = []
        self.detection_success_count = 0
        self.detection_fail_count = 0
        print("✓ Statistics reset")
    
    def set_threshold(self, threshold: float):
        """Set the angle threshold for poor posture detection."""
        if 5 <= threshold <= 45:
            self.threshold = threshold
            print(f"✓ Threshold set to {threshold}°")
        else:
            print(f"⚠️ Threshold must be between 5° and 45°. Keeping {self.threshold}°")
    
    def release(self):
        """Release resources."""
        if self.detector is not None:
            self.detector.close()
        print("✓ Resources released")