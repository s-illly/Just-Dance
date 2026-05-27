import cv2

def run_end_screen(player_cap, total_score, best_combo):
    """ Show final score and wait for q or space"""
    if total_score >= 4000:
        letter = "S"
        colour = (147, 255, 147) # green
    elif total_score >= 3000:
        letter = "A"
        colour = (147, 220, 255)    # yellow
    elif total_score >= 2000:
        letter = "B"
        colour = (147, 147, 255)    # orange
    elif total_score >= 1000:
        letter = "C"
        colour = (80, 80, 220)      # red
    else:
        letter = "D"
        colour = (120, 120, 120)    # grey
    
    while True:
        ret, frame = player_cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Letter grade (big, centred)
        cv2.putText(frame, letter, (w//2 - 40, h//2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, colour, 6)

        # Score
        cv2.putText(frame, f"Score: {int(total_score)}", (w//2 - 130, h//2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Best combo
        cv2.putText(frame, f"Best combo: x{best_combo}", (w//2 - 130, h//2 + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)

        # Prompt
        cv2.putText(frame, "SPACE to play again   Q to quit",
                    (w//2 - 220, h//2 + 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 140, 140), 1)

        cv2.imshow("Just Dance", frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' '):
            return True # play again
        if key == ord('q'):
            return False # quit 
