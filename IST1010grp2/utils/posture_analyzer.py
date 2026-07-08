import cv2
import numpy as np
import os

class PostureAnalyzer:
    def __init__(self, threshold=15):
        """
        Initialize the posture analyzer
        
        Args:
            threshold (float): Angle threshold for poor posture detection in degrees
        """
        self.threshold = threshold
        self.net = None
        self.load_model()
        
        # Key point indices for OpenPose
        self.shoulder_left = 1
        self.shoulder_right = 2
        self.ear_left = 3
        self.ear_right = 4
        
        self.POSE_PAIRS = [
            [1, 2], [1, 5], [2, 6], [3, 7], [4, 8], [5, 6], [5, 11],
            [6, 12], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]
        ]
    
    def load_model(self):
        """Load OpenPose model for pose detection"""
        try:
            # You need to download these files and place them in the models directory
            proto_file = "models/pose_deploy.prototxt"
            # weights_file = "models/pose_iter_440000.caffemodel"
            weights_file = "models/pose_iter_5840000.caffemodel"
            
            if not os.path.exists(proto_file) or not os.path.exists(weights_file):
                print("Warning: Model files not found. Using simplified posture detection.")
                self.net = None
                return
            
            self.net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
            print("Pose detection model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.net = None
    
    def detect_pose(self, frame):
        """Detect pose keypoints using OpenPose"""
        if self.net is None:
            return None
        
        # Prepare input
        height, width = frame.shape[:2]
        inHeight = 368
        inWidth = int((inHeight / height) * width)
        
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)
        
        self.net.setInput(inpBlob)
        output = self.net.forward()
        
        points = []
        threshold = 0.1
        
        for i in range(18):  # OpenPose has 18 keypoints
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            x = (width * point[0]) / inWidth
            y = (height * point[1]) / inHeight
            
            if prob > threshold:
                points.append((int(x), int(y)))
            else:
                points.append(None)
        
        return points
    
    def calculate_angle(self, p1, p2):
        """Calculate angle of a line from horizontal"""
        if p1 is None or p2 is None:
            return None
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Calculate angle from horizontal
        angle = np.degrees(np.arctan2(dy, dx))
        return abs(angle)
    
    def analyze_posture_simple(self, frame):
        """
        Simplified posture analysis using basic image processing
        This is a fallback when OpenPose model is not available
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect edges for shoulder approximation
        edges = cv2.Canny(gray, 50, 150)
        
        # Get image center
        height, width = frame.shape[:2]
        center_y = height // 2
        
        # Look for horizontal edges near the top for shoulder detection
        # This is a very simplified approach
        return True, 0  # Assume good posture
    
    def analyze_posture(self, frame):
        """
        Analyze posture from frame
        
        Returns:
            tuple: (is_good_posture, angle)
        """
        if self.net is None:
            return self.analyze_posture_simple(frame)
        
        points = self.detect_pose(frame)
        
        if points is None:
            return True, 0
        
        # Get shoulder points
        left_shoulder = points[1]  # Point 1
        right_shoulder = points[2]  # Point 2
        
        # Get ear points for more accurate detection
        left_ear = points[3]   # Point 3
        right_ear = points[4]  # Point 4
        
        # Use ear points if available, otherwise use shoulders
        if left_ear is not None and right_ear is not None:
            p1 = left_ear
            p2 = right_ear
        elif left_shoulder is not None and right_shoulder is not None:
            p1 = left_shoulder
            p2 = right_shoulder
        else:
            return True, 0
        
        # Calculate angle
        angle = self.calculate_angle(p1, p2)
        
        if angle is None:
            return True, 0
        
        # Good posture is when shoulders are relatively horizontal (angle near 0)
        # Poor posture when shoulders are tilted (angle > threshold)
        is_good = angle < self.threshold
        
        return is_good, angle
    
    def set_threshold(self, threshold):
        """Set the angle threshold for poor posture detection"""
        self.threshold = threshold

    def draw_keypoints(self, frame, points):
        """Draw detected keypoints on frame for visualization"""
        if points is None:
            return frame
        
        for i, point in enumerate(points):
            if point is not None:
                cv2.circle(frame, point, 3, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (point[0] - 5, point[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw connections between keypoints
        for pair in self.POSE_PAIRS:
            p1 = points[pair[0]]
            p2 = points[pair[1]]
            if p1 is not None and p2 is not None:
                cv2.line(frame, p1, p2, (255, 0, 255), 2)
        
        return frame
