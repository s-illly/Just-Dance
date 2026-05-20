import cv2 
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

SCORING_JOINTS = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE
]

def landmark_to_pixel(landmark, frame_w, frame_h):
    """Convert normalized (0-1) landmark coords to pixel coords"""
    return (int(landmark.x * frame_w), int(landmark.y * frame_h))

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
    

with mp_pose.Pose(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            # mp_drawing.draw_landmarks(
            #     frame,
            #     results.pose_landmarks,
            #     mp_pose.POSE_CONNECTIONS, # connecting joints
            #     landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style()
            # )
            lm = results.pose_landmarks.landmark
            for joint in SCORING_JOINTS:
                landmark = lm[joint]
                if landmark.visibility < 0.5:
                    continue
                px, py = landmark_to_pixel(landmark, w, h)

                cv2.circle(frame, (px, py), 8, (147, 112, 219), -1)
                cv2.circle(frame, (px, py), 8, (255,255,255), 1)
            # left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            # right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            # left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

            # print(f"Left shoulder  x={left_shoulder.x:.3f}  y={left_shoulder.y:.3f}  visibility={left_shoulder.visibility:.2f}")
            # print(f"Right shoulder x={right_shoulder.x:.3f}  y={right_shoulder.y:.3f}")
            # print(f"Left wrist     x={left_wrist.x:.3f}  y={left_wrist.y:.3f}")
            # print("---")

        cv2.imshow("Pose detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()