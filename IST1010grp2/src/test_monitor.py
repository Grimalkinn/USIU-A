"""
Main Posture Monitor Application
Uses posture_analyzer.py for posture detection
"""

import cv2
import time
import pygame
import threading
import sys
from test_analyzer import PostureAnalyzer

class PostureMonitor:
    """
    Main application class for posture monitoring
    """
    
    def __init__(self, alert_sound_path: str = "sounds/alert.mp3", 
                 threshold: float = 15.0):
        """
        Initialize the posture monitor
        
        Args:
            alert_sound_path: Path to alert sound file
            threshold: Angle threshold for poor posture detection
        """
        try: pygame.mixer.music.load(self.alert_sound_path)
        except Exception as e: pass
        # Initialize analyzer
        self.analyzer = PostureAnalyzer(threshold=threshold)
        
        # Camera
        self.cap = None
        self.camera_id = 0
        
        # Alert settings
        self.alert_sound_path = alert_sound_path
        self.alert_delay = 3  # seconds
        self.bad_posture_start_time = None
        self.alert_playing = False
        
        # State
        self.is_running = False
        
        # Initialize pygame for sound
        self._init_sound()
        
        print("✓ PostureMonitor initialized")
    
    def _init_sound(self):
        """Initialize pygame sound system"""
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(self.alert_sound_path)
            print("✓ Sound system initialized")
        except Exception as e:
            print(f"⚠️ Sound initialization failed: {e}")
            print("   Posture alerts will be visual only")
    
    def play_alert(self):
        """Play alert sound in background thread"""
        if not self.alert_playing and pygame.mixer.get_init():
            self.alert_playing = True
            threading.Thread(target=self._play_sound_thread, daemon=True).start()
    
    def _play_sound_thread(self):
        """Thread function for playing sound"""
        try:
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Error playing sound: {e}")
        finally:
            self.alert_playing = False
    
    def start(self, camera_id: int = 0):
        """
        Start the posture monitor
        
        Args:
            camera_id: Camera device ID
        """
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera {camera_id}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        self.bad_posture_start_time = None
        
        print(f"✓ Camera {camera_id} opened successfully")
        print("\n=== Posture Monitor Controls ===")
        print("  'q' - Quit")
        print("  'r' - Reset statistics")
        print("  '+' - Increase threshold by 1°")
        print("  '-' - Decrease threshold by 1°")
        print("  's' - Show statistics")
        print("  'd' - Toggle debug mode")
        print("  'h' - Show help")
        print("==============================\n")
        
        return True
    
    def run(self):
        """Main monitoring loop"""
        if not self.is_running:
            print("❌ Monitor not started. Call start() first.")
            return
        
        print("✓ Monitoring started!")
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ Failed to grab frame")
                break
            
            # Analyze posture - FIXED: Accept 3 return values
            posture_status, angle, landmarks = self.analyzer.analyze_posture(frame)
            
            # Handle alert
            if not posture_status:
                if self.bad_posture_start_time is None:
                    self.bad_posture_start_time = time.time()
                elif time.time() - self.bad_posture_start_time > self.alert_delay:
                    self.play_alert()
                    # Reset timer to prevent alert spam
                    self.bad_posture_start_time = time.time()
            else:
                self.bad_posture_start_time = None
            
            # Draw overlays
            frame = self.analyzer.draw_posture_info(frame, posture_status, angle, landmarks)
            frame = self.analyzer.draw_angle_visualization(frame, angle)
            
            # Add controls help
            cv2.putText(frame, "q:Quit r:Reset +/-:Threshold s:Stats d:Debug", 
                       (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show frame
            cv2.imshow('Posture Monitor', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop()
                break
            elif key == ord('r'):
                self.analyzer.reset_statistics()
                self.bad_posture_start_time = None
                print("✓ Statistics reset")
            elif key == ord('+'):
                self.analyzer.set_threshold(self.analyzer.threshold + 1)
            elif key == ord('-'):
                self.analyzer.set_threshold(self.analyzer.threshold - 1)
            elif key == ord('s'):
                stats = self.analyzer.get_statistics()
                print("\n=== Posture Statistics ===")
                print(f"Total Frames: {stats['total_frames']}")
                print(f"Good Frames: {stats['good_frames']}")
                print(f"Bad Frames: {stats['bad_frames']}")
                print(f"Good Percentage: {stats['good_percentage']:.1f}%")
                print(f"Overall Score: {stats['score']:.1f}%")
                print(f"Average Angle: {stats['average_angle']:.1f}°")
                print(f"FPS: {stats['fps']:.1f}")
                print(f"Detection Success Rate: {stats.get('detection_success_rate', 0):.1f}%")
                print("==========================\n")
            elif key == ord('d'):
                self.analyzer.debug_mode = not self.analyzer.debug_mode
                print(f"Debug mode: {'ON' if self.analyzer.debug_mode else 'OFF'}")
            elif key == ord('h'):
                print("\n=== Help ===")
                print("q: Quit")
                print("r: Reset statistics")
                print("+: Increase threshold")
                print("-: Decrease threshold")
                print("s: Show statistics")
                print("d: Toggle debug mode")
                print("h: Show this help")
                print("============\n")
    
    def stop(self):
        """Stop the monitor and cleanup"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.analyzer.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        print("\n✓ Monitor stopped")


def main():
    """Main entry point"""
    print("=== Posture Monitor v2.0 (OpenPose) ===\n")
    
    # Check if model files exist
    import os
    if not os.path.exists("models/pose_deploy.prototxt"):
        print("⚠️ Warning: pose_deploy.prototxt not found in models/")
    if not os.path.exists("models/pose_iter_584000.caffemodel"):
        print("⚠️ Warning: pose_iter_584000.caffemodel not found in models/")
        print("   Please download the model file:")
        print("   https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe-models/raw/master/caffe_models/openpose/caffe_model/pose_iter_584000.caffemodel?raw=true")
    
    # Create monitor with default settings
    monitor = PostureMonitor(
        alert_sound_path="sounds/alert.mp3",
        threshold=15.0
    )
    
    # Start with camera 0
    if not monitor.start(camera_id=0):
        print("Failed to start monitor")
        return
    
    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()