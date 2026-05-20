import cv2
import mediapipe as mp
import numpy as np
import os 
import sys

mp_pose = mp.solutions.pose 

def extract_keypoints(results):
    """
    Turn a Mediapipe results object into a (33, 4) numpy array
    each row is [x,y,z,visibility] for one landmark
    return 0 is no pose is detected
    """
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
    else:
        return np.zeros((33, 4))
   
def extract_video_poses(video_path, output_dir = "poses", show_preview=True):
    """
    Run MediaPipe on every frame of a video and save keypoints to disk.

    Output: poses/<video_name>.npy  shape = (num_frames, 33, 4)
    Also saves metadata (fps, frame count) to poses/<video_name>_meta.npy
    """
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

    all_keypoints = []
    
    with mp_pose.Pose(
        static_image_mode = False,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    ) as pose:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            keypoints = extract_keypoints(results)
            all_keypoints.append(keypoints)
            
            # Progress bar 
            if frame_idx % 100 == 0:
                pct = (frame_idx / total) * 100
                print(f"  [{pct:5.1f}%]  frame {frame_idx}/{total}", end="\r")

            # Small preview window
            if show_preview:
                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )
                small = cv2.resize(frame, (480, 270))
                cv2.imshow("Extracting poses", small)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped early by user")
                    break
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    pose_array = np.stack(all_keypoints)
    np.save(output_path, pose_array)
    meta = np.array([fps, len(all_keypoints)])
    np.save(meta_path, meta)
    print(f"\nDone. Saved {len(all_keypoints)} frames → {output_path}")
    print(f"Array shape: {pose_array.shape}")

    return output_path


if __name__ == "__main__":
    video = sys.argc[1] if len(sys.argv) > 1 else "videos/test_dance.mp4" 
    extract_video_poses(video)
