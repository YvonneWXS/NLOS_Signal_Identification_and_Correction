# shift_detector.py — CUSUM Distribution Shift Detection
# ========================================================
# Detects persistent changes in positioning innovation sequence.
# When the scene transitions (e.g., entering/leaving building canyon),
# CUSUM triggers a reset of the adaptive thresholds.
#
# Reference: classic CUSUM (Cumulative Sum) control chart.
# Also compatible with River's ADWIN for online mean-change detection.
# ================================================================

import numpy as np


class CUSUMShiftDetector:
    """CUSUM control chart for detecting distribution shifts in innovation.
    
    Tracks two one-sided CUSUM statistics:
      - Positive CUSUM: detects upward shift (MoG getting worse)
      - Negative CUSUM: detects downward shift (MoG getting better)
    
    When either exceeds threshold, signals a scene change and resets.
    """

    def __init__(self, target=0.0, allowance=20.0, threshold=100.0):
        """
        Args:
            target: expected mean innovation (m, 0 = MoG same as LS)
            allowance: slack before CUSUM accumulates (m)
            threshold: detection threshold (m)
        """
        self.target = target
        self.allowance = allowance
        self.threshold = threshold
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.shift_detected = False
        self.detection_history = []
        self.innovation_buffer = []

    def update(self, innovation_meters):
        """Update CUSUM statistics and return shift detection result.
        
        Returns: 'NONE', 'POSITIVE' (MoG worsening), or 'NEGATIVE' (MoG improving)
        """
        self.innovation_buffer.append(innovation_meters)
        if len(self.innovation_buffer) > 100:
            self.innovation_buffer.pop(0)

        # Positive CUSUM: detects upward shift
        self.cusum_pos = max(0, self.cusum_pos +
                             innovation_meters - self.target - self.allowance)
        # Negative CUSUM: detects downward shift
        self.cusum_neg = max(0, self.cusum_neg -
                             innovation_meters + self.target - self.allowance)

        shift = 'NONE'
        if self.cusum_pos > self.threshold:
            shift = 'POSITIVE'
            self.cusum_pos = 0.0
            self.shift_detected = True
        elif self.cusum_neg > self.threshold:
            shift = 'NEGATIVE'
            self.cusum_neg = 0.0
            self.shift_detected = True

        self.detection_history.append(shift)
        return shift

    def should_recompute(self):
        """True if a shift was detected and thresholds should be re-learned."""
        if self.shift_detected:
            self.shift_detected = False
            return True
        return False

    def get_statistics(self):
        return {
            'cusum_pos': float(self.cusum_pos),
            'cusum_neg': float(self.cusum_neg),
            'detection_count': len([x for x in self.detection_history if x != 'NONE']),
            'recent_innovation_mean': float(np.mean(self.innovation_buffer[-20:]))
                if len(self.innovation_buffer) >= 20 else 0.0,
        }


class ADWINShiftDetector:
    """ADWIN-based shift detector using River library.
    
    ADWIN (Adaptive Windowing) maintains a variable-length window of recent 
    observations. When the window shrinks significantly, a distribution change
    is detected. More precise than CUSUM for detecting arbitrary shifts.
    """
    
    def __init__(self, clock=32, max_buckets=5, min_window_length=5,
                 grace_period=10, delta=0.002):
        try:
            from river.drift import ADWIN
            self.adwin = ADWIN(
                clock=clock, max_buckets=max_buckets,
                min_window_length=min_window_length,
                grace_period=grace_period, delta=delta,
            )
            self.has_river = True
        except ImportError:
            self.has_river = False
            self.adwin = None
            self.fallback_buffer = []
            self.fallback_mean = 0.0
        
        self.shift_count = 0

    def update(self, innovation_meters):
        """Update ADWIN and return whether drift was detected."""
        if not self.has_river:
            # Fallback: simple running mean threshold
            self.fallback_buffer.append(innovation_meters)
            if len(self.fallback_buffer) > 50:
                self.fallback_buffer.pop(0)
            old_mean = self.fallback_mean
            new_mean = np.mean(self.fallback_buffer[-20:]) if len(self.fallback_buffer) >= 20 else 0
            self.fallback_mean = np.mean(self.fallback_buffer)
            if abs(new_mean - old_mean) > 30 and old_mean != 0:
                self.shift_count += 1
                return True
            return False
        
        _ = self.adwin.update(innovation_meters)
        if self.adwin.drift_detected:
            self.shift_count += 1
            return True
        return False

    def should_recompute(self):
        """True if a drift was detected recently."""
        return False  # ADWIN doesn't need explicit recompute; it adapts internally

    def get_statistics(self):
        if self.has_river:
            return {
                'shift_count': self.shift_count,
                'window_width': self.adwin.width if self.adwin else 0,
            }
        return {'shift_count': self.shift_count, 'fallback': True}


print("Module 3 shift_detector.py loaded — CUSUMShiftDetector, ADWINShiftDetector ready.")
