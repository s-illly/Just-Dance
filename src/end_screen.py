import cv2
from hud import PLAYER_COLOURS

def run_end_screen(player_cap, scores, best_combos):
    """
    Show final scores for all players and wait for space or q.
    scores:      list of total scores, one per player
    best_combos: list of best combo counts, one per player
    """
    best_score = max(scores)

    if best_score >= 4000:
        letter = "S"
        letter_colour = (147, 255, 147)
    elif best_score >= 3000:
        letter = "A"
        letter_colour = (147, 220, 255)
    elif best_score >= 2000:
        letter = "B"
        letter_colour = (147, 147, 255)
    elif best_score >= 1000:
        letter = "C"
        letter_colour = (80, 80, 220)
    else:
        letter = "D"
        letter_colour = (120, 120, 120)

    while True:
        ret, frame = player_cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # letter grade (big, centred at top)
        cv2.putText(frame, letter, (w // 2 - 40, h // 2 - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, letter_colour, 6)

        # per-player scores listed below the grade
        for i, (score, best) in enumerate(zip(scores, best_combos)):
            colour = PLAYER_COLOURS[i % len(PLAYER_COLOURS)]
            y = h // 2 + i * 45
            cv2.putText(frame, f"P{i + 1}  {int(score)} pts   best combo x{best}",
                        (w // 2 - 200, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

        # prompt
        cv2.putText(frame, "SPACE to play again   Q to quit",
                    (w // 2 - 220, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 140, 140), 1)

        cv2.imshow("Just Dance", frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' '):
            return True
        if key == ord('q'):
            return False
