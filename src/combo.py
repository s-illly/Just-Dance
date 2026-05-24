class ComboTracker:
    """
    Tracks consecutive good frames and computes a score multiplier.

    Combo tiers:
      0-9   hits → 1x
      10-24 hits → 2x
      25-49 hits → 4x
      50+   hits → 8x  (max)
    """
    TIERS = [
        (50, 8),
        (25, 4),
        (10, 2),
        (0, 1),
    ]
    def __init__(self):
        self.combo = 0
        self.best = 0
        self.multiplier = 1

    def update(self, grade):
        if grade in ("PERFECT", "GOOD"):
            self.combo += 1
            self.best = max(self.best, self.combo)
        elif grade == ("MISS"):
            self.combo = 0

        for threshold, mult in self.TIERS:
            if self.combo >= threshold:
                self.multiplier = mult
                break 
        
        return self.multiplier 
    
    def is_on_fire(self):
        """ True when on max multiplier """
        return self.multiplier == 8
        
