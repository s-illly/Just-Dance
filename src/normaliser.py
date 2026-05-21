import numpy as np

LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

def normalise_pose(keypoints):
    """
    Normalise a (33, 4) keypoint array so scores are fair regardless
    of the player's size or distance from the camera.

    Steps:
      1. Find the hip midpoint and subtract it from every landmark
         (now the hips sit at the origin)
      2. Measure torso height (hips to shoulders) and divide every
         coordinate by it (now all poses have the same scale)

    Returns a (33, 3) array of normalised [x, y, z] — visibility
    is dropped since we only use it for filtering, not math.
    """
    # drop visibility column
    coords = keypoints[:, :3].copy()
    left_hip = coords[LEFT_HIP]
    right_hip = coords[RIGHT_HIP]
    hip_center = (left_hip + right_hip) / 2
    # origin
    coords -= hip_center

    left_shoulder = coords[LEFT_SHOULDER]
    right_shoulder = coords[RIGHT_SHOULDER]
    shoulder_center = (left_shoulder + right_shoulder) / 2
    
    # Euclidean distance from hip (origin)
    torso_height = np.linalg.norm(shoulder_center)

    if torso_height < 1e-6:
        # no person detected
        return np.zeros((33, 3))
    
    coords /= torso_height
    return coords 
