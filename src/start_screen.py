import cv2
import numpy as np

def draw_start_screen(frame, song_name="Test Dance"):
    """
    Draw a start screen
    """
    h, w = frame.shape[:2]

    overlay = frame.copy()
    # Title
    cv2.putText(frame, "JUST DANCE", (w//2 - 200, h//2 - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)

    # Song name
    cv2.putText(frame, song_name, (w//2 - 150, h//2 - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)

    # Prompt
    cv2.putText(frame, "Press SPACE to dance", (w//2 - 180, h//2 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 220, 100), 2)

    # Controls reminder
    cv2.putText(frame, "Q to quit", (w//2 - 70, h//2 + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1)

    return frame

def run_start_screen(player_cap, song_name="Test Dance"):
    """
    Show start screen on webcam feed until player presses space
    """
    while True:
        ret, frame = player_cap.read()
        if not ret:
            return False
        key = cv2.waitKey(30) & 0xFF

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1280, 480))
        frame = draw_start_screen(frame, song_name)

        cv2.imshow("Just Dance", frame)
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord(' '):
            return True
        if key == ord('q'):
            return False
