import cv2
import mediapipe as mp
import numpy as np 
import pygame 
import time 
import os
import sys 
import subprocess
import tempfile 
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode
from extractor import landmarks_to_keypoints, ensure_model, _hip_x
from scorer import scorer
from hud import draw_grade_banner, draw_player_scores
from end_screen import run_end_screen
from start_screen import run_start_screen, draw_start_screen
from combo import ComboTracker

sys.path.insert(0, os.path.dirname(__file__))

PANEL_W, PANEL_H = 640, 480
SCORE_STRIP_H = 120
LAG_FRAMES = 3

# one colour per player slot (BGR)
PLAYER_COLOURS = [
    (50, 220, 50),   # P1: green
    (50, 50, 220),   # P2: red
    (220, 200, 0),   # P3: cyan
    (0, 165, 255),   # P4: orange
]

POSE_CONNECTIONS = [
    # (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8), # face
    # (9,10), # mouth
    (11,12), # shoulders
    (11,13),(13,15),(15,17),(15,19),(15,21),(17,19), # left arm & hand
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20), # right arm & hand
    (11,23),(12,24),(23,24), # torso
    (23,25),(25,27),(27,29),(27,31),(29,31), # left leg
    (24,26),(26,28),(28,30),(28,32),(30,32), # right leg
]

def draw_skeleton(frame, landmarks, colour):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(pts) and end_idx < len(pts):
            cv2.line(frame, pts[start_idx], pts[end_idx], colour, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, colour, -1)


def extract_audio(video_path):
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run([
       "ffmpeg", "-y", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", tmp 
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp 

def run_game(video_path, poses_path, title, num_players = 1, model_path=None):
    """
    video_path: path to dance video (.mp4)
    poses_path: path to extracted keypoints (.npy) shape: (frames, num_dancers, 33, 4)
    num_players: how many people are playing (1-4)
    """

    model_path = ensure_model(model_path)

    # assets 
    song_name = title
    ref_poses = np.load(poses_path)
    meta = np.load(poses_path.replace(".npy", "_meta.npy"))
    fps, total_frames = meta
    num_ref_dancers = ref_poses.shape[1]
    print(f"Loaded {int(total_frames)} frames @ {fps:.1f}fps | " 
          f"{num_ref_dancers} reference dancer(s) | {num_players} player(s)")

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
    
    # game loop
    options = PoseLandmarkerOptions(
        base_options = BaseOptions(model_asset_path=model_path),
        running_mode = RunningMode.VIDEO,
        num_poses = num_players,
        min_pose_detection_confidence = 0.5,
        min_pose_presence_confidence = 0.5,
        min_tracking_confidence = 0.5,
    )

    play_again = True
    while play_again:
        # start screen
        if not run_start_screen(player_cap, song_name):
            break

        # game state 
        scores = [0.0] * num_players
        combos = [ComboTracker() for _ in range(num_players)]
        grade_alphas = [0.0] * num_players
        current_grades = ["MISS"] * num_players

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
        frame_idx = 0
        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pygame.mixer.music.play()
        cam_timestamp_ms = 0

        # main loop 
        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                #loop_start = time.time()

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

                # send to mediapipe , detect all players
                detect_frame = cv2.resize(player_frame, (320, 240))
                rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, cam_timestamp_ms)
                cam_timestamp_ms += 1

                # draw skeleton over player
                detected = sorted(result.pose_landmarks or [], key=_hip_x)

                # score 
                score_idx = max(0, frame_idx - LAG_FRAMES)

                # draw players, score against reference dancer
                for i, landmarks in enumerate(detected[:num_players]):
                    draw_skeleton(player_frame, landmarks, PLAYER_COLOURS[i % len(PLAYER_COLOURS)])

                    # draw hud 
                    player_kp = landmarks_to_keypoints(landmarks)
                    ref_dancer_idx = i % num_ref_dancers

                    ref_kp = ref_poses[score_idx, ref_dancer_idx]
                    frame_score, grade = scorer(ref_kp, player_kp)

                    # composite 
                    multiplier = combos[i].update(grade)
                    scores[i] += frame_score * multiplier 
                    current_grades[i] = grade 
                    if grade in ("PERFECT", "GOOD"):
                        grade_alphas[i] = 1.0
                    grade_alphas[i] = max(0.0, grade_alphas[i] - 0.033)


                # score strip 
                score_strip = draw_player_scores(
                    scores = scores,
                    grades = current_grades, 
                    multipliers = [c.multiplier for c in combos],
                    combo_counts = [c.combo for c in combos],
                    num_players = num_players,
                    active_count = len(detected[:num_players]),
                    width = PANEL_W * 2,
                    height = SCORE_STRIP_H
                )
                top_row = np.hstack([ref_frame, player_frame])
                combined = np.vstack([top_row, score_strip])
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
            
        # end screen 
        pygame.mixer.music.stop()
        if play_again:
            play_again = run_end_screen(player_cap, scores, [c.best for c in combos])
        
    ref_cap.release()
    player_cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()
    os.remove(tmp_audio)

    print(f"\nGame over! Final score: {[int(s) for s in scores]}")
    return [int(s) for s in scores]
            