#!/usr/bin/env python3

import cv2

# print(f"OpenCV version: {cv2.__version__}")
# print(f"DNN module available: {hasattr(cv2, 'dnn')}")
# if hasattr(cv2, 'dnn'): print(f"DNN attributes: {[attr for attr in dir(cv2.dnn) if not attr.startswith('_')]}")


print(f"OpenCV version: {cv2.__version__}")
try:
    net = cv2.dnn.readNetFromCaffe("models/pose_deploy.prototxt", "models/pose_iter_584000.caffemodel")
    print("✓ DNN works!")
except Exception as e: print(f"✗ Error: {e}")