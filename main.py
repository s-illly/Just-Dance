import sys 
import os 

sys.path.insert(0, "src")
from game import run_game

VIDEO = "src/videos/test_dance.mp4"
POSES = "src/poses/test_dance.npy"

if not os.path.exists(VIDEO):
    print(f"Video not found: {VIDEO}")
    print("Run: yt-dlp -o videos/test_dance.mp4 <URL>")
    sys.exit(1)
if not os.path.exists(POSES):
    print(f"Poses not found: {POSES}")
    print("Run: python src/extractor.py videos/test_dance.mp4")
    sys.exit(1)

run_game(VIDEO, POSES)
