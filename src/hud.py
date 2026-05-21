import cv2 
import numpy as np

GRADE_COLOURS = {
    "PERFECT": (147, 255, 147), # green
    "GOOD": (147, 220, 255), # yellow
    "OK": (147, 147, 255), # orange
    "MISS": (80,  80,  220), # red
}

def draw_score_bar(frame, score, grade, total_score):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w, 60), (0,0,0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    # score bar background
    bar_x, bar_y, bar_w, bar_h = 20, 15, 300, 22
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)

    # score bar fill 
    fill_w = int(bar_w * score)
    colour = GRADE_COLOURS.get(grade, (150, 150, 150))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), colour, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)

    # Grade text
    cv2.putText(frame, grade, (bar_x + bar_w + 12, bar_y + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)

    # Total score (top right)
    score_text = f"{int(total_score)}"
    cv2.putText(frame, score_text, (w - 120, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, "pts", (w - 48, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    return frame

def draw_grade_banner(frame, grade, alpha):
    """
    Flash a big grade banner in the centre of the frame.
    alpha fades from 1.0 → 0.0 over ~30 frames.
    """
    if alpha <= 0 or grade == "MISS":
        return frame

    h, w = frame.shape[:2]
    colour = GRADE_COLOURS.get(grade, (255, 255, 255))

    # Scale text
    font_scale = 2.5 if grade == "PERFECT" else 2.0
    thickness  = 4

    (text_w, text_h), _ = cv2.getTextSize(grade, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = (w - text_w) // 2
    ty = h // 2 + text_h // 2

    # Draw with fading opacity via a blended overlay
    overlay = frame.copy()
    cv2.putText(overlay, grade, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, thickness)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return frame

def draw_countdown(frame, seconds_left):
    """Show a countdown before the song starts."""
    h, w = frame.shape[:2]
    text = str(seconds_left) if seconds_left > 0 else "GO!"
    colour = (100, 220, 100) if seconds_left == 0 else (255, 255, 255)

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 6)
    cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 4.0, colour, 6)
    return frame

