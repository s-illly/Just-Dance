import numpy as np
from normaliser import normalise_pose

SCORING_JOINTS = [
    # parent, child, weight, label
    (11, 13, 2.0, "left upper arm"),                                                                          
    (12, 14, 2.0, "right upper arm"),                                                                         
    (13, 15, 3.0, "left lower arm"),                                                                          
    (14, 16, 3.0, "right lower arm"),                                                                         
    (23, 25, 1.0, "left thigh"),                                                                              
    (24, 26, 1.0, "right thigh"),                                                                             
    (25, 27, 0.8, "left shin"),                                                                               
    (26, 28, 0.8, "right shin"),
]


GRADE_PERFECT = 0.95
GRADE_GOOD = 0.88
GRADE_OK = 0.75

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
    
    weighted_sum = 0.0
    weight_total = 0.0

    for parent, child, weight, _label in SCORING_JOINTS:
        ref_vis = min(ref_keypoints[parent, 3], ref_keypoints[child, 3])
        player_vis = min(player_keypoints[parent, 3], player_keypoints[child, 3])
        if ref_vis < 0.5 or player_vis < 0.5:
            continue 

        ref_bone = ref_norm[child] - ref_norm[parent]
        player_bone = player_norm[child] - player_norm[parent]

        sim = cosine_similarity(ref_bone, player_bone)
        score = (sim + 1) / 2
        weighted_sum += score * weight 
        weight_total += weight 

    if weight_total == 0:
        return 0.0, "MISS"
    final_score = weighted_sum / weight_total 
    if final_score >= GRADE_PERFECT:
        return final_score, "PERFECT"
    elif final_score >= GRADE_GOOD:
        return final_score, "GOOD"
    elif final_score >= GRADE_OK:
        return final_score, "OK"   
    return 0.0, "MISS"
