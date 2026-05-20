from importlib.metadata import version as pkg_version
import cv2 
import mediapipe as mp
import numpy as np
import scipy 
import yt_dlp 

print(f"OpenCV:    {cv2.__version__}")
print(f"MediaPipe: {mp.__version__}")
print(f"NumPy:     {np.__version__}")
print(f"pygame:    {pkg_version('pygame')}")
print(f"scipy:     {scipy.__version__}")
print(f"yt-dlp:    {yt_dlp.version.__version__}")
print("\nAll packages imported successfully!")