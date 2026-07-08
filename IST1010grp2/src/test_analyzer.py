"""
Posture Analyzer Module - Debug Version
Uses OpenPose (Caffe model) for pose detection
"""

import cv2
import numpy as np
import math
import time
from typing import Tuple, Optional, Dict, List
import os

class PostureAnalyzer:
    """
    Analyzes sitting posture using OpenPose Caffe model.
    """
    
    def __init__(self, threshold: float = 15.0, 
            model_proto: str = "models/pose_deploy.prototxt", 
            # model_weights: str = "models/pose_iter_584000.caffemodel"):
            model_weights: str = "models/pose_iter_440000.caffemodel"):

        """
        Initialize the posture analyzer with OpenPose model.
        """
        self.threshold = threshold
        self.net = None
        self.input_width = 368
        self.input_height = 368
        self.debug_mode = False  # Enable debug output
        
        # COCO keypoint mapping
        self.KEYPOINT_INDICES = {
            'left_shoulder': 5,
            'right_shoulder': 2,
            'left_ear': 17,
            'right_ear': 16,
            'left_hip': 11,
            'right_hip': 8,
            'left_elbow': 6,
            'right_elbow': 3,
            'left_wrist': 7,
            'right_wrist': 4,
            'nose': 0,
            'neck': 1,
        }
        
        # Statistics
        self.total_frames = 0
        self.good_posture_frames = 0
        self.bad_posture_frames = 0
        self.posture_history = []
        self.max_history = 100
        self.processing_times = []
        self.fps = 0
        
        # Debug counters
        self.detection_success_count = 0
        self.detection_fail_count = 0
        
        # Load the model
        self._load_model(model_proto, model_weights)
    
    def _load_model(self, proto_path: str, weights_path: str):
        """Load the OpenPose Caffe model."""
        print("\n=== Loading OpenPose Model ===")
        
        # Check if files exist
        if not os.path.exists(proto_path):
            print(f"❌ Prototxt file not found at: {proto_path}")
            print(f"   Current directory: {os.getcwd()}")
            return
            
        if not os.path.exists(weights_path):
            print(f"❌ Caffe model file not found at: {weights_path}")
            print(f"   File size should be ~99.9 MB")
            return
        
        # Check file sizes
        proto_size = os.path.getsize(proto_path) / 1024  # KB
        weights_size = os.path.getsize(weights_path) / (1024 * 1024)  # MB
        print(f"✓ Prototxt file found: {proto_size:.1f} KB")
        print(f"✓ Weights file found: {weights_size:.1f} MB")
        
        if weights_size < 90:
            print(f"⚠️ Warning: Weights file seems too small ({weights_size:.1f} MB).")
            print("   It might be the Git LFS pointer file, not the actual model.")
            print("   Please download the actual model file.")
        
        try:
            self.net = cv2.dnn.readNetFromCaffe(proto_path, weights_path)
            if self.net is None: print("✓ OpenPose model not loaded!!")
            else: print("✓ OpenPose model loaded successfully!")
            
            # Optional: Enable GPU if available
            # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.net = None
        
        print("==============================\n")
    
    def _detect_keypoints(self, frame: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        Detect keypoints in the frame using OpenPose.
        """
        if self.net is None:
            if self.debug_mode:
                print("⚠️ Network not loaded!")
            return None
        
        h, w = frame.shape[:2]
        
        # Prepare input blob
        inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, 
                                         (self.input_width, self.input_height),
                                         (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(inp_blob)
        
        # Forward pass
        try:
            output = self.net.forward()
        except Exception as e:
            print(f"❌ Forward pass error: {e}")
            return None
        
        # Process output
        points = []
        threshold_confidence = 0.1
        
        # Debug: Print output shape
        if self.debug_mode and self.total_frames % 30 == 0:
            print(f"Output shape: {output.shape}")
            print(f"Output min/max: {output.min():.3f} / {output.max():.3f}")
        
        for i in range(18):  # COCO has 18 keypoints
            prob_map = output[0, i, :, :]
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
            
            x = (w * point[0]) / self.input_width
            y = (h * point[1]) / self.input_height
            
            if prob > threshold_confidence:
                points.append((int(x), int(y)))
                if self.debug_mode and self.total_frames % 30 == 0:
                    print(f"  Keypoint {i}: ({int(x)}, {int(y)}) confidence: {prob:.3f}")
            else:
                points.append(None)
        
        # Count detected keypoints
        detected = sum(1 for p in points if p is not None)
        if self.debug_mode and self.total_frames % 30 == 0:
            print(f"Detected {detected}/18 keypoints")
        
        if detected < 4:
            if self.debug_mode:
                print(f"⚠️ Only {detected} keypoints detected. Need at least 4 for posture analysis.")
            self.detection_fail_count += 1
            return None
        
        self.detection_success_count += 1
        return points
    
    def analyze_posture(self, frame: np.ndarray) -> Tuple[bool, float, Optional[Dict]]:
        """
        Analyze posture from a single frame.
        """
        start_time = time.time()
        
        # Detect keypoints
        keypoints = self._detect_keypoints(frame)
        
        # Default values
        is_good = True
        angle = 0.0
        landmarks = None
        
        if keypoints:
            # Extract relevant landmarks
            landmarks = self._extract_landmarks(keypoints)
            
            # Check if we have the necessary landmarks
            if landmarks.get('left_shoulder') and landmarks.get('right_shoulder'):
                # Calculate shoulder angle
                angle = self._calculate_shoulder_angle(landmarks)
                
                # Determine if posture is good
                is_good = angle <= self.threshold
                
                if self.debug_mode and self.total_frames % 30 == 0:
                    print(f"Left Shoulder: {landmarks.get('left_shoulder')}")
                    print(f"Right Shoulder: {landmarks.get('right_shoulder')}")
                    print(f"Angle: {angle:.1f}°, Threshold: {self.threshold}°")
                    print(f"Posture: {'Good' if is_good else 'Bad'}")
            else:
                if self.debug_mode and self.total_frames % 30 == 0:
                    print("⚠️ Shoulder landmarks not detected!")
                self.detection_fail_count += 1
        else:
            if self.debug_mode and self.total_frames % 30 == 0:
                print("⚠️ No keypoints detected!")
            self.detection_fail_count += 1
        
        # Update statistics
        self.total_frames += 1
        if is_good:
            self.good_posture_frames += 1
        else:
            self.bad_posture_frames += 1
            
        # Update history
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
    
    def _extract_landmarks(self, keypoints: List) -> Dict[str, Tuple[int, int]]:
        """Extract relevant landmarks from detected keypoints."""
        landmarks = {}
        for name, idx in self.KEYPOINT_INDICES.items():
            if idx < len(keypoints):
                landmarks[name] = keypoints[idx]
        return landmarks
    
    def _calculate_shoulder_angle(self, landmarks: Dict) -> float:
        """Calculate shoulder angle from landmarks."""
        try:
            left_shoulder = landmarks.get('left_shoulder')
            right_shoulder = landmarks.get('right_shoulder')
            
            if left_shoulder is None or right_shoulder is None:
                return 0.0
            
            lx, ly = left_shoulder
            rx, ry = right_shoulder
            
            # Calculate angle from horizontal
            dx = rx - lx
            dy = ry - ly
            
            # Avoid division by zero
            if dx == 0:
                return 90.0
            
            angle_rad = math.atan2(abs(dy), abs(dx))
            angle_deg = math.degrees(angle_rad)
            
            return angle_deg
            
        except Exception as e:
            print(f"Error calculating angle: {e}")
            return 0.0
    
    def draw_keypoints(self, frame: np.ndarray, keypoints: List) -> np.ndarray:
        """Draw detected keypoints on the frame for debugging."""
        if keypoints is None:
            return frame
        
        for i, point in enumerate(keypoints):
            if point is not None:
                # Draw different colors for different body parts
                if i in [2, 5]:  # Shoulders
                    color = (0, 255, 255)  # Yellow
                    radius = 5
                elif i in [3, 4, 6, 7]:  # Arms
                    color = (255, 0, 255)  # Magenta
                    radius = 3
                elif i in [8, 11]:  # Hips
                    color = (255, 255, 0)  # Cyan
                    radius = 4
                else:
                    color = (0, 255, 0)  # Green
                    radius = 3
                
                cv2.circle(frame, point, radius, color, -1)
                cv2.putText(frame, str(i), (point[0]-10, point[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def draw_posture_info(self, frame: np.ndarray, is_good: bool, angle: float, 
                          landmarks: Optional[Dict] = None) -> np.ndarray:
        """Draw posture information on the frame."""
        h, w = frame.shape[:2]
        
        # Status box (top-left)
        cv2.rectangle(frame, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 150), (255, 255, 255), 2)
        
        status = "✓ GOOD POSTURE" if is_good else "✗ BAD POSTURE"
        color = (0, 255, 0) if is_good else (0, 0, 255)
        cv2.putText(frame, status, (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(frame, f"Angle: {angle:.1f}°", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Threshold: {self.threshold}°", (20, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Detection stats
        if self.detection_success_count + self.detection_fail_count > 0:
            success_rate = (self.detection_success_count / 
                          (self.detection_success_count + self.detection_fail_count)) * 100
            cv2.putText(frame, f"Detection: {success_rate:.0f}%", (20, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Statistics box (top-right)
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
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        fill_width = int((score / 100) * bar_width)
        color = (0, 255, 0) if score > 70 else (0, 255, 255) if score > 40 else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     color, -1)
        
        total = self.total_frames
        good = self.good_posture_frames
        cv2.putText(frame, f"Good: {good}  Bad: {total - good}  Total: {total}", 
                   (w - 290, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def draw_angle_visualization(self, frame: np.ndarray, angle: float) -> np.ndarray:
        """Draw a gauge showing current angle."""
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
                'detection_success_rate': 0.0
            }
        
        avg_angle = 0
        if self.posture_history:
            avg_angle = sum(p['angle'] for p in self.posture_history) / len(self.posture_history)
        
        good_pct = (self.good_posture_frames / self.total_frames) * 100
        success_rate = (self.detection_success_count / 
                       (self.detection_success_count + self.detection_fail_count)) * 100 if (self.detection_success_count + self.detection_fail_count) > 0 else 0
        
        return {
            'total_frames': self.total_frames,
            'good_frames': self.good_posture_frames,
            'bad_frames': self.bad_posture_frames,
            'good_percentage': good_pct,
            'bad_percentage': 100 - good_pct,
            'score': self.calculate_posture_score(),
            'average_angle': avg_angle,
            'fps': self.fps,
            'detection_success_rate': success_rate
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
        """Release any resources."""
        print("✓ Resources released")


# Test function with debugging
def test_posture_analyzer():
    """
    Test the posture analyzer with webcam and extensive debugging.
    """
    import cv2
    
    print("\n=== Posture Analyzer Debug Test ===\n")
    print("This will help identify why detection isn't working.")
    print("\nMake sure you have:")
    print("1. Good lighting")
    print("2. Full upper body visible")
    print("3. Camera at eye level")
    print("4. Model files in the 'models' folder")
    print("\nPress 'q' to quit")
    print("Press 'd' to toggle debug mode")
    
    # Initialize analyzer with debug mode on
    analyzer = PostureAnalyzer(threshold=15.0)
    
    if analyzer.net is None:
        print("\n❌ Model not loaded. Please check:")
        print("  - File paths in models/ folder")
        print("  - File sizes (weights should be ~99.9 MB)")
        print("  - You have the actual model file, not the LFS pointer")
        return
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return
    
    print("\n✓ Camera opened. Starting detection...\n")
    
    debug = True
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        
        # Analyze posture
        is_good, angle, landmarks = analyzer.analyze_posture(frame)
        
        # Create a copy for visualization with keypoints
        vis_frame = frame.copy()
        
        # Try to get keypoints for visualization
        keypoints = analyzer._detect_keypoints(frame) if analyzer.net else None
        if keypoints:
            vis_frame = analyzer.draw_keypoints(vis_frame, keypoints)
        
        # Draw posture info
        vis_frame = analyzer.draw_posture_info(vis_frame, is_good, angle, landmarks)
        vis_frame = analyzer.draw_angle_visualization(vis_frame, angle)
        
        # Add debug info
        if debug:
            stats = analyzer.get_statistics()
            cv2.putText(vis_frame, f"Detection Success: {stats['detection_success_rate']:.0f}%", 
                       (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(vis_frame, f"Keypoints Detected: {sum(1 for p in keypoints if p is not None) if keypoints else 0}/18", 
                       (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show angle calculation visualization
            if landmarks and landmarks.get('left_shoulder') and landmarks.get('right_shoulder'):
                lx, ly = landmarks['left_shoulder']
                rx, ry = landmarks['right_shoulder']
                cv2.line(vis_frame, (lx, ly), (rx, ry), (0, 255, 255), 2)
                mid_x = (lx + rx) // 2
                mid_y = (ly + ry) // 2
                cv2.putText(vis_frame, f"Angle: {angle:.1f}°", (mid_x - 30, mid_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Posture Analyzer Debug', vis_frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug = not debug
            print(f"Debug mode: {'ON' if debug else 'OFF'}")
        elif key == ord('r'):
            analyzer.reset_statistics()
        elif key == ord('+'):
            analyzer.set_threshold(analyzer.threshold + 1)
        elif key == ord('-'):
            analyzer.set_threshold(analyzer.threshold - 1)
    
    cap.release()
    analyzer.release()
    cv2.destroyAllWindows()
    print("\n✓ Test completed")


if __name__ == "__main__":
    test_posture_analyzer()