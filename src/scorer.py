import numpy as np
from normaliser import normalise_pose

SCORING_JOINTS = [
    11, 12, # shoulders 
    13, 14, # elbows 
    15, 16, # wrists 
    23, 24, # hips 
    25, 26, # knees 
]

GRADE_PERFECT = 0.92 
GRADE_GOOD = 0.75
GRADE_OK = 0.55

def cosine_similarity(a, b):
    """
    How similar are vectors a and b? Returns 1.0 (identical) to -1.0 (opposite)
    """
    dot = np.dot(a,b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
    return dot / (norm_a * norm_b)


def scorer(ref_keypoints, player_keypoints):
    """
    Compare one frame of reference pose vs player pose.

    Args:
        ref_keypoints: (33, 4) array from .npy file
        player_keypoints: (33, 4) array from mp

    Returns: 
        score (float): 0.0-1.0
        grade (str): 'PERFECT', 'GOOD', 'OK', or 'MISS'
    """
    
    ref_norm = normalise_pose(ref_keypoints)
    player_norm = normalise_pose(player_keypoints)

    if np.all(ref_norm == 0) or np.all(player_norm == 0):
        return 0.0, "MISS"
    
    joint_scores = []
    for joint_idx in SCORING_JOINTS:
        # x,y,z
        ref_vec = ref_norm[joint_idx]
        player_vec = player_norm[joint_idx]

        ref_visible = ref_keypoints[joint_idx, 3]
        player_visible = player_keypoints[joint_idx, 3]
        if ref_visible < 0.5 or player_visible < 0.5:
            continue # skip joint 

        sim = cosine_similarity(ref_vec, player_vec)
        score = (sim + 1) / 2
        joint_scores.append(score)

    if not joint_scores:
        return 0.0, "MISS"
    final_score = float(np.mean(joint_scores))
    if final_score >= GRADE_PERFECT:
        return final_score, "PERFECT"
    elif final_score >= GRADE_GOOD:
        return final_score, "GOOD"
    elif final_score >= GRADE_OK:
        return final_score, "OK"   
    return final_score, "MISS"
