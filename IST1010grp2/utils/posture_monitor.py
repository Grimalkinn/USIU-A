import sys
import cv2
import numpy as np
import time
import pygame
import threading
from datetime import datetime
import os
from posture_analyzer import PostureAnalyzer

class PostureMonitor:
    def __init__(self, alert_sound_path="sounds/alert.wav"):
        """
        Initialize the Posture Monitor System
        
        Args:
            alert_sound_path (str): Path to the alert sound file
        """
        if len(sys.argv) > 1: self.sysPlat = sys.argv[1]
        self.cap = None
        self.analyzer = PostureAnalyzer()
        self.alert_threshold = 15  # degrees tolerance
        self.is_running = False
        self.bad_posture_start_time = None
        self.alert_delay = 3  # seconds before alert
        self.alert_playing = False
        
        # Initialize pygame for sound
        pygame.mixer.init()
        self.alert_sound = alert_sound_path
        
        # Statistics
        self.total_frames = 0
        self.bad_posture_frames = 0
        self.good_posture_frames = 0
        
    def load_alert_sound(self):
        """Load the alert sound file"""
        try:
            pygame.mixer.music.load(self.alert_sound)
            return True
        except Exception as e:
            print(f"Error loading sound: {e}")
            return False
    
    def play_alert(self):
        """Play the alert sound in a separate thread"""
        if not self.alert_playing:
            self.alert_playing = True
            threading.Thread(target=self._play_sound_thread, daemon=True).start()
    
    def _play_sound_thread(self):
        """Thread function to play sound without blocking"""
        try:
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Error playing sound: {e}")
        finally:
            self.alert_playing = False
    
    def start_monitoring(self, camera_id=0):
        """Start the posture monitoring system"""
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return False
        
        if not self.load_alert_sound():
            print("Warning: Alert sound not loaded")
        
        self.is_running = True
        self.bad_posture_start_time = None
        print("Posture Monitoring Started. Press 'q' to quit.")
        print("Press 'r' to reset statistics.")
        
        return True
    
    def draw_info(self, frame, posture_status, angle):
        """Draw information on the frame"""
        height, width = frame.shape[:2]
        
        # Status box
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 120), (255, 255, 255), 2)
        
        # Status text
        status_text = "Good Posture" if posture_status else "Bad Posture"
        color = (0, 255, 0) if posture_status else (0, 0, 255)
        cv2.putText(frame, f"Status: {status_text}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Angle: {angle:.1f}°", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Alert: {'ON' if not posture_status else 'OFF'}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 0, 255) if not posture_status else (255, 255, 255), 2)
        
        # Statistics
        cv2.rectangle(frame, (width - 300, 10), (width - 10, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (width - 300, 10), (width - 10, 120), (255, 255, 255), 2)
        
        good_percentage = (self.good_posture_frames / max(1, self.total_frames)) * 100
        cv2.putText(frame, f"Good Posture: {good_percentage:.1f}%", 
                   (width - 290, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Bad Posture: {100 - good_percentage:.1f}%", 
                   (width - 290, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Frames: {self.total_frames}", 
                   (width - 290, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main loop for posture monitoring"""
        if not self.is_running:
            print("Please start monitoring first")
            return
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Analyze posture
            posture_status, angle = self.analyzer.analyze_posture(frame)
            self.total_frames += 1
            
            if posture_status:
                self.good_posture_frames += 1
                self.bad_posture_start_time = None
            else:
                self.bad_posture_frames += 1
                if self.bad_posture_start_time is None:
                    self.bad_posture_start_time = time.time()
                elif time.time() - self.bad_posture_start_time > self.alert_delay:
                    self.play_alert()
                    self.bad_posture_start_time = time.time()  # Reset to prevent spam
            
            # Draw information on frame
            frame = self.draw_info(frame, posture_status, angle)
            
            # Show frame
            cv2.imshow('Posture Monitor', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop()
                break
            elif key == ord('r'):
                self.reset_statistics()
                print("Statistics reset")
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.total_frames = 0
        self.bad_posture_frames = 0
        self.good_posture_frames = 0
        self.bad_posture_start_time = None
    
    def stop(self):
        """Stop the monitoring system"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Posture Monitoring Stopped")
    
    def set_threshold(self, threshold):
        """Set the angle threshold for poor posture detection"""
        if 5 <= threshold <= 45:
            self.alert_threshold = threshold
            self.analyzer.set_threshold(threshold)
            print(f"Threshold set to {threshold} degrees")
        else:
            print("Threshold must be between 5 and 45 degrees")

def main():
    """Main function to run the posture monitor"""
    monitor = PostureMonitor()
    
    if not monitor.start_monitoring():
        return
    
    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        monitor.stop()

if __name__ == "__main__":
    main()
