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
num_dancers = int(input("How many dancers to track in the video? (1-4): "))
num_players = int(input("How many players will be playing? (1-4): "))

try:
    tmp_dir = tempfile.mkdtemp()
    POSES = extract_video_poses(VIDEO, output_dir = tmp_dir, num_dancers = num_dancers)
    run_game(VIDEO, POSES, info['title'], num_players = num_players)
finally:
    os.remove(VIDEO)
    shutil.rmtree(tmp_dir)
