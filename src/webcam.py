import cv2 
import numpy as np
import sys 


# video_path = sys.argv[1] if len(sys.argv) > 1 else 0
# cap = cv2.VideoCapture(video_path)

# fps = cap.get(cv2.CAP_PROP_FPS)
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# print(f"Video: {w}x{h} @ {fps:.1f}fps, {total_frames} frames")
# frame_idx = 0

cap = cv2.VideoCapture(0)

TARGET_W, TARGET_H = 640, 480

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break
    frame = cv2.flip(frame, 1)
    player_panel = cv2.resize(frame, (TARGET_W, TARGET_H))

    ref_panel = np.zeros((TARGET_H, TARGET_W, 3), dtype = np.uint8)
    ref_panel[:] = (40, 40, 40)
    cv2.putText(ref_panel, "Reference video", (160, TARGET_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (180,180,180), 2)

    # concatenates two same height frames into one wide image 
    combined = np.hstack([ref_panel, player_panel])
    cv2.imshow("Just Dance", combined)
    
    # frame_idx += 1
    
    # frame = cv2.flip(frame, 1) # 1 = horizontal flip
    # h, w = frame.shape[:2] # frame 0 = height , 1 = width , 2 = channels 

    # # Drawing rectangle 
    # # cv2.rectangle(img, top left, bottom right, colour BGR, thickness)
    # cv2.rectangle(frame, (20, 20), (200, 70), (0,0,0), -1) # black filled box 
    # cv2.rectangle(frame, (20, 20), (200, 70), (255, 255, 255), 1) # white filled border


    # # Writing text
    # # cv2.putText(img, text, origin, font, scale, color BGR, thickness)
    # cv2.putText(frame, "Score: 0", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # # Drawing circle
    # center = (w // 2, h // 2)
    # cv2.circle(frame, center, 40, (0,255,255), 3)

    # # Drawing lines 
    # cv2.line(frame, (0,h//2), (w,h//2), (0,255,0), 1)

    # # Semi transparent
    # overlay = frame.copy()
    # cv2.rectangle(overlay, (0, h-60), (w,h), (0,0,0), -1)
    # alpha = 0.5 # 0 = invisible,  1 = opaque 
    # frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

    # cv2.putText(frame, "Press Q to quit", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # cv2.putText(frame, f"Frame {frame_idx}/{total_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # cv2.imshow("Video playback", frame)

    # delay = max(1, int(1000 / fps))
    if cv2.waitKey(1) & 0xFF == ord('q'): # quit key is q
        break

cap.release()
cv2.destroyAllWindows()