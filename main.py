import sys 
import os, tempfile 
import shutil 

sys.path.insert(0, "src")
from game import run_game
from video import download_video
from extractor import extract_video_poses

url = input("Paste a youtube url: ").strip()
VIDEO, info = download_video(url)
print(f"\nSaved to: {VIDEO}")

try:
    tmp_dir = tempfile.mkdtemp()
    POSES = extract_video_poses(VIDEO, output_dir = tmp_dir)
    run_game(VIDEO, POSES, info['title'])
finally:
    os.remove(VIDEO)
    shutil.rmtree(tmp_dir)
