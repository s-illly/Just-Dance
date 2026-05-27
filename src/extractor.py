import cv2
import mediapipe as mp
import numpy as np
import os 
import sys
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode
import urllib.request

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"             
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pose_landmarker_full.task", # slower more accurate model 
)

def ensure_model(path=None):
    if path is None:
        path = _DEFAULT_MODEL
    if not os.path.exists(path):
        print(f"Downloading pose landmarker model to {path}")
        urllib.request.urlretrieve(_MODEL_URL, path)
        print("Done")
    return path

def landmarks_to_keypoints(landmarks):
    """ Convert Tasks API landmark list to (33, 4) [x,y,z, visibility] array"""
    return np.array([
        [lm.x, lm.y, lm.z, lm.visibility if lm.visibility is not None else 1.0]
        for lm in landmarks
    ])

def _hip_x(landmarks):
    """ Define origin point """
    return (landmarks[23].x + landmarks[24].x) / 2

def extract_video_poses(video_path, output_dir = "poses", num_dancers = 1, model_path=None):
    """
    Run PoseLandmarker on every frame of a video and save keypoints to disk
    num_dancers : how many dancers to track in the video (1-4), sorted left to right
                0 is leftmost dancer 
    output: <output_dir>/<name>.npy shape = (num_frames, num_dancers, 33, 4)
            <output_dir>/<name>_meta.npy [fps, frame_count]
    """
    model_path = ensure_model(model_path)
    os.makedirs(output_dir, exist_ok=True)
    # "test_dance"
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # all poses from all frames 
    # [frame count, 33, 4]
    output_path = os.path.join(output_dir, f"{video_name}.npy")
    # video fps and frame count 
    meta_path = os.path.join(output_dir, f"{video_name}_meta.npy")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    options = PoseLandmarkerOptions(
        base_options = BaseOptions(model_asset_path=model_path),
        running_mode = RunningMode.VIDEO,
        num_poses = num_dancers,
        min_pose_detection_confidence = 0.5,
        min_pose_presence_confidence = 0.5,
        min_tracking_confidence = 0.5,
    )
    all_keypoints = []
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            timestamp_ms = int(frame_idx * 1000.0 / fps) if fps > 0 else frame_idx
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_poses = np.zeros((num_dancers, 33, 4))
            detected = sorted(result.pose_landmarks or [], key = _hip_x)
            for i, lms in enumerate(detected[:num_dancers]):
                frame_poses[i] = landmarks_to_keypoints(lms)
            all_keypoints.append(frame_poses)
            
            # Progress bar 
            if frame_idx % 100 == 0:
                pct = (frame_idx / total) * 100 if total > 0 else 0
                print(f"  [{pct:5.1f}%]  frame {frame_idx}/{total}", end="\r")

            # Small preview window
            # if show_preview:
            #     if results.pose_landmarks:
            #         mp.solutions.drawing_utils.draw_landmarks(
            #             frame,
            #             results.pose_landmarks,
            #             mp_pose.POSE_CONNECTIONS
            #         )
            #     small = cv2.resize(frame, (480, 270))
            #     cv2.imshow("Extracting poses", small)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         print("\nStopped early by user")
            #         break
            frame_idx += 1

    cap.release()

    pose_array = np.stack(all_keypoints)
    np.save(output_path, pose_array) # frames, num_dancers, 33, 4
    np.save(meta_path, np.array([fps, len(all_keypoints)]))
    print(f"\nDone. Saved {len(all_keypoints)} frames → {output_path}")
    print(f"Array shape: {pose_array.shape}")

    return output_path

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "videos/test_dance.mp4" 
    dancers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    extract_video_poses(video, num_dancers = dancers)
