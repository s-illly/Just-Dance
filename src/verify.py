import numpy as np
import sys 
import os 
from scorer import scorer 

path = sys.argv[1] if len(sys.argv) > 1 else "poses/test_dance.npy"
poses = np.load(path)
meta = np.load(path.replace(".npy", "_meta.npy"))

# fps, num_frames = meta
# print(f"Shape:      {poses.shape}")
# print(f"Frames:     {int(num_frames)}")
# print(f"FPS:        {fps:.1f}")
# print(f"Duration:   {num_frames / fps:.1f}s")
# print()

# left_shoulder = poses[0, 11]
# print(f"Frame 0 · left shoulder:")
# print(f"  x={left_shoulder[0]:.4f}  y={left_shoulder[1]:.4f}  z={left_shoulder[2]:.4f}  visibility={left_shoulder[3]:.4f}")

# # check how many empty frames 
# empty_frames = np.all(poses == 0, axis=(1, 2)).sum()
# print(f"\nFrames with no detection: {empty_frames}/{int(num_frames)} ({100*empty_frames/num_frames:.1f}%)")

# sys.path.insert(0, os.path.dirname(__file__))
# compare frame against itself
ref = poses[100]
score, grade = scorer(ref, ref)
print(f"Same frame vs itself:  score={score:.3f}  grade={grade}")

# compare frame with nearby frame 
ref2 = poses[175]
score2, grade2 = scorer(ref, ref2)
print(f"Frame 100 vs 175:      score={score2:.3f}  grade={grade2}")

# compare against 0
ref4 = np.zeros((33, 4))
score4, grade4 = scorer(ref, ref4)
print(f"Frame vs no detection: score={score4:.3f}  grade={grade4}")
