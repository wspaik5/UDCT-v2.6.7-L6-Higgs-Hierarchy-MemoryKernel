"""
memory_kernel.py

Adaptive Memory Kernel for stabilizing Monte Carlo time series.
This module implements a robust memory-assisted smoothing algorithm
designed for lattice gauge theory simulations, particularly for
Higgs hierarchy stabilization studies (UDCT v2.6.7).

Key features:
- Welford's online algorithm for numerically stable mean/variance updates
- Median-based outlier rejection to protect memory from large fluctuations
- Adaptive exponential moving average with warm-up phase
- Phase jump detection for tracking sudden shifts in the underlying signal

Author: Won Shik Paik
"""

import numpy as np
from numba import njit


@njit
def welford_update(existing_aggregate, new_value):
    """
    Perform one step of Welford's online algorithm for numerically stable
    calculation of mean and variance.

    Parameters
    ----------
    existing_aggregate : tuple
        (count, mean, M2) from previous steps
    new_value : float
        New data point to incorporate

    Returns
    -------
    tuple
        Updated (count, mean, M2)
    """
    (count, mean, M2) = existing_aggregate
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return (count, mean, M2)


@njit
def calculate_variance(existing_aggregate):
    """
    Calculate variance from Welford aggregate.
    Returns 0.0 if count < 2.
    """
    (count, mean, M2) = existing_aggregate
    if count < 2:
        return 0.0
    return M2 / (count - 1)


@njit
def memory_kernel(raw_series, alpha=0.12, outlier_threshold=0.008,
                  large_diff_threshold=0.003, warm_up_steps=20):
    """
    Apply Adaptive Memory Kernel to a raw time series.

    The kernel combines:
    1. Outlier rejection (protects memory from extreme spikes)
    2. Welford-based online statistics for stability
    3. Adaptive exponential smoothing with warm-up phase
    4. Phase jump detection (tracks genuine shifts in the signal)

    Parameters
    ----------
    raw_series : np.ndarray
        1D array of raw Monte Carlo measurements (e.g., plaquette values)
    alpha : float
        Base smoothing factor for exponential moving average (default: 0.12)
    outlier_threshold : float
        Threshold for detecting outliers (default: 0.008)
    large_diff_threshold : float
        Threshold for detecting phase jumps (default: 0.003)
    warm_up_steps : int
        Number of initial steps to use higher alpha for faster adaptation

    Returns
    -------
    np.ndarray
        Smoothed time series after applying the memory kernel
    """
    n = len(raw_series)
    smoothed = np.empty(n)
    smoothed[0] = raw_series[0]

    # Initialize Welford aggregates
    agg = (1, raw_series[0], 0.0)

    for t in range(1, n):
        current = raw_series[t]
        previous_smoothed = smoothed[t - 1]

        # Calculate deviation from current memory state
        diff = abs(current - previous_smoothed)

        # === Outlier Rejection ===
        if diff > outlier_threshold:
            # Treat as outlier: do not update memory aggressively
            smoothed[t] = previous_smoothed
            continue

        # === Phase Jump Detection ===
        if diff > large_diff_threshold:
            # Genuine shift detected → increase responsiveness temporarily
            effective_alpha = min(0.5, alpha * 3.0)
        else:
            effective_alpha = alpha

        # === Warm-up Phase (faster adaptation at the beginning) ===
        if t < warm_up_steps:
            effective_alpha = max(effective_alpha, 0.3)

        # === Welford Update + Exponential Smoothing ===
        agg = welford_update(agg, current)
        variance = calculate_variance(agg)

        # Adaptive smoothing
        smoothed[t] = (1 - effective_alpha) * previous_smoothed + effective_alpha * current

    return smoothed
