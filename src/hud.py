import cv2
import numpy as np

PLAYER_COLOURS = [
    (50, 220, 50), # P1: green
    (50, 50, 220), # P2: red
    (220, 200, 0), # P3: cyan
    (0, 165, 255), 1# P4: orange
]

_SCORE_CEIL = 5000 

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

def draw_combo(frame, multiplier, combo_count):
    """ Draw combo mult in the bottom left corner """
    if multiplier <= 1:
        return frame
    
    h, w = frame.shape[:2]
    colours = {
        2: (147, 220, 255),    # yellow
        4: (80,  147, 255),    # orange
        8: (80,  80,  220),    # red — on fire
    }
    colour = colours.get(multiplier, (255, 255, 255))
    
    # Multiplier badge
    badge = f"x{multiplier}"
    cv2.putText(frame, badge, (20, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, colour, 3)

    # Combo count underneath
    cv2.putText(frame, f"{combo_count} combo", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)

    # Pulsing ring when on fire (x8)
    if multiplier == 8:
        cv2.circle(frame, (42, h - 58), 38, colour, 2)

    return frame


def draw_player_scores(scores, grades, multipliers, combo_counts, num_players, active_count, width, height):
    """
    Build the score strip shown below the main video panels.

    Each player gets an equal-width slot showing:
      - coloured player label + current grade
      - a bar that fills as their total score grows toward _SCORE_CEIL
      - total score number
      - combo multiplier if active

    Slots for players not currently detected on camera are dimmed.
    """
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    strip[:] = (20, 20, 20)

    slot_w = width // num_players

    for i in range(num_players):
        x = i * slot_w
        is_active = i < active_count
        colour = PLAYER_COLOURS[i % len(PLAYER_COLOURS)]
        label_colour = colour if is_active else tuple(c // 3 for c in colour)

        # player label
        cv2.putText(strip, f"P{i + 1}", (x + 10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_colour, 2)

        # current grade
        grade = grades[i]
        grade_colour = GRADE_COLOURS.get(grade, (100, 100, 100)) if is_active else (50, 50, 50)
        cv2.putText(strip, grade, (x + 65, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, grade_colour, 2)

        # score bar — fills toward _SCORE_CEIL
        bar_x, bar_y = x + 10, 45
        bar_w, bar_h = slot_w - 20, 16
        cv2.rectangle(strip, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill = int(bar_w * min(scores[i] / _SCORE_CEIL, 1.0))
        bar_colour = GRADE_COLOURS.get(grade, (100, 100, 100)) if is_active else (40, 40, 40)
        if fill > 0:
            cv2.rectangle(strip, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_colour, -1)
        cv2.rectangle(strip, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)

        # total score
        score_colour = (200, 200, 200) if is_active else (80, 80, 80)
        cv2.putText(strip, f"{int(scores[i])} pts", (x + 10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_colour, 1)

        # combo — only shown when multiplier > 1 and player is on screen
        if multipliers[i] > 1 and is_active:
            cv2.putText(strip, f"x{multipliers[i]}  {combo_counts[i]} combo",
                        (x + 10, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)

        # divider line between slots
        if i < num_players - 1:
            cv2.line(strip, (x + slot_w, 5), (x + slot_w, height - 5), (60, 60, 60), 1)

    return strip
