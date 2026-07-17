
"""
Memory Kernel for Higgs Hierarchy Stabilization
UDCT v2.6.7 - Adaptive Threshold Version

This module implements an adaptive memory kernel with:
- Welford's online algorithm for numerical stability
- Dynamic threshold based on running standard deviation
- Outlier rejection using median
- Phase jump fast adaptation
"""

import numpy as np
from collections import deque


class MemoryKernel:
    """
    Adaptive Memory Kernel with dynamic threshold and phase jump detection.
    Designed for long-running Monte Carlo simulations (e.g., Higgs hierarchy studies).
    """

    def __init__(self, 
                 memory_length: int = 20,
                 decay_factor: float = 0.85,
                 outlier_threshold: float = 0.008,
                 large_diff_threshold: float = 0.007,
                 min_large_diff_threshold: float = 0.0025,
                 adaptive_threshold_factor: float = 2.8,
                 phase_jump_adaptation_steps: int = 8):
        
        self.memory_length = memory_length
        self.decay_factor = decay_factor
        self.outlier_threshold = outlier_threshold
        self.large_diff_threshold = large_diff_threshold
        self.min_large_diff_threshold = min_large_diff_threshold
        self.adaptive_threshold_factor = adaptive_threshold_factor
        self.phase_jump_adaptation_steps = phase_jump_adaptation_steps

        # Internal states
        self.history = deque(maxlen=memory_length)
        self.smoothed_output = None
        self.last_smoothed = None
        
        # Welford's algorithm variables
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        
        # Phase jump adaptation
        self.adaptation_counter = 0
        self.current_decay = decay_factor

    def _update_welford(self, value: float):
        """Update running mean and variance using Welford's algorithm."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    def _get_std(self) -> float:
        """Return current standard deviation."""
        if self.count < 2:
            return 0.0
        return np.sqrt(self.M2 / (self.count - 1))

    def _get_dynamic_large_diff_threshold(self) -> float:
        """Compute adaptive threshold for phase jump detection."""
        current_std = self._get_std()
        dynamic_threshold = current_std * self.adaptive_threshold_factor
        return max(self.min_large_diff_threshold, dynamic_threshold)

    def apply(self, raw_value: float) -> float:
        """
        Apply memory kernel to a new raw value.
        Returns the smoothed output.
        """
        # Warm-up phase
        if len(self.history) < self.memory_length:
            self.history.append(raw_value)
            self._update_welford(raw_value)
            self.smoothed_output = raw_value
            self.last_smoothed = raw_value
            return raw_value

        # Update history
        self.history.append(raw_value)
        self._update_welford(raw_value)

        # Outlier detection using median
        median = np.median(self.history)
        if abs(raw_value - median) > self.outlier_threshold:
            # Treat as outlier: return previous smoothed value
            return self.last_smoothed

        # Phase jump detection
        dynamic_threshold = self._get_dynamic_large_diff_threshold()
        diff = abs(raw_value - self.last_smoothed)

        if diff > dynamic_threshold:
            # Activate fast adaptation
            self.adaptation_counter = self.phase_jump_adaptation_steps
            self.current_decay = 0.55  # temporarily faster adaptation
        else:
            if self.adaptation_counter > 0:
                self.adaptation_counter -= 1
            else:
                self.current_decay = self.decay_factor

        # Exponential smoothing
        if self.smoothed_output is None:
            self.smoothed_output = raw_value
        else:
            self.smoothed_output = (self.current_decay * self.smoothed_output +
                                    (1 - self.current_decay) * raw_value)

        self.last_smoothed = self.smoothed_output
        return self.smoothed_output

    def get_stats(self):
        """Return current statistics for debugging."""
        return {
            'mean': self.mean,
            'std': self._get_std(),
            'smoothed': self.smoothed_output,
            'adaptation_active': self.adaptation_counter > 0
        }
