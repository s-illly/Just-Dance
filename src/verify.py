import numpy as np
import sys 

path = sys.argv[1] if len(sys.argv) > 1 else "poses/test_dance.npy"
poses = np.load(path)
meta = np.load(path.replace(".npy", "_meta.npy"))

fps, num_frames = meta
print(f"Shape:      {poses.shape}")
print(f"Frames:     {int(num_frames)}")
print(f"FPS:        {fps:.1f}")
print(f"Duration:   {num_frames / fps:.1f}s")
print()

left_shoulder = poses[0, 11]
print(f"Frame 0 · left shoulder:")
print(f"  x={left_shoulder[0]:.4f}  y={left_shoulder[1]:.4f}  z={left_shoulder[2]:.4f}  visibility={left_shoulder[3]:.4f}")

# check how many empty frames 
empty_frames = np.all(poses == 0, axis=(1, 2)).sum()
print(f"\nFrames with no detection: {empty_frames}/{int(num_frames)} ({100*empty_frames/num_frames:.1f}%)")