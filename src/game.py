import cv2
import mediapipe as mp
import numpy as np 
import pygame 
import time 
import os
import sys 
import subprocess
import tempfile 
from extractor import extract_keypoints
from scorer import scorer
from hud import draw_countdown, draw_grade_banner, draw_score_bar

sys.path.insert(0, os.path.dirname(__file__))

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 

PANEL_W, PANEL_H = 640, 480
LAG_FRAMES = 3
COUNTDOWN_SEC = 3

def extract_audio(video_path):
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run([
       "ffmpeg", "-y", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", tmp 
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp 

def run_game(video_path, poses_path):
    """
    video_path: path to dance video (.mp4)
    poses_path: path to extracted keypoints (.npy)
    """

    # assets 
    ref_poses = np.load(poses_path)
    meta = np.load(poses_path.replace(".npy", "_meta.npy"))
    fps, total_frames = meta
    frame_delay = max(1, int(1000/fps))
    print(f"Loaded {int(total_frames)} reference frames @ {fps:.1f}fps")

    # audio 
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    tmp_audio = extract_audio(video_path)
    pygame.mixer.music.load(tmp_audio)

    # video
    ref_cap = cv2.VideoCapture(video_path)
    player_cap = cv2.VideoCapture(0)

    if not player_cap.isOpened():
        print("ERROR: could not open webcam")
        return
    
    # game state 
    total_score = 0
    frame_idx = 0
    grade_alpha = 0.0
    current_grade = "MISS"

    # countdown
    # for i in range(COUNTDOWN_SEC, -1, -1):
    #     ret, player_frame = player_cap.read()
    #     if not ret:
    #         break
    #     player_frame = cv2.flip(player_frame, 1)
    #     player_frame = cv2.resize(player_frame, (PANEL_W * 2, PANEL_H))
    #     player_frame = draw_countdown(player_frame, i)
    #     cv2.imshow("Just Dance", player_frame)
    #     cv2.waitKey(1000)

    # start audio
    pygame.mixer.music.play()
    game_start = time.time()

    # main loop 
    with mp_pose.Pose(
        model_complexity = 0, #lighter model 
        smooth_landmarks = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    ) as pose:
        while True:
            loop_start = time.time()

            # read reference video frame 
            ret_ref, ref_frame = ref_cap.read()
            if not ret_ref or frame_idx >= int(total_frames):
                break
            ref_frame = cv2.resize(ref_frame, (PANEL_W, PANEL_H))

            # read player video frame
            ret_player, player_frame = player_cap.read()
            if not ret_player:
                break
            player_frame = cv2.flip(player_frame, 1)
            player_frame = cv2.resize(player_frame, (PANEL_W, PANEL_H))

            # send to mediapipe 
            detect_frame = cv2.resize(player_frame, (320, 240))
            rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # draw skeleton over player
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    player_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS, # connecting joints
                )

            # score 
            score_idx = max(0, frame_idx - LAG_FRAMES)
            ref_kp = ref_poses[score_idx]
            player_kp = extract_keypoints(results)
            frame_score, grade = scorer(ref_kp, player_kp)
            total_score += frame_score * 10 

            # grade banner 
            if grade in ("PERFECT", "GOOD"):
                grade_alpha = 1.0
                current_grade = grade 
            grade_alpha = max(0.0, grade_alpha - 0.033)

            # draw hud 
            ref_panel = draw_score_bar(ref_frame, frame_score, current_grade, total_score)
            player_panel = draw_grade_banner(player_frame, current_grade, grade_alpha)

            # composite 
            combined = np.hstack([ref_panel, player_panel])

            # show
            cv2.imshow("Just Dance", combined)

            # timing 
            audio_sec = pygame.mixer.music.get_pos() / 1000.0
            target_frame = int(audio_sec * fps)

            # skip frames if lagging behind 
            if target_frame > frame_idx + 1:
                ref_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                frame_idx = target_frame 
            else: 
                frame_idx += 1

            # elapsed_ms = (time.time() - loop_start) * 1000
            # wait_ms = max(1, frame_delay - int(elapsed_ms))
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        
    ref_cap.release()
    player_cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()
    os.remove(tmp_audio)

    print(f"\nGame over! Final score: {int(total_score)}")
    return int(total_score)
        