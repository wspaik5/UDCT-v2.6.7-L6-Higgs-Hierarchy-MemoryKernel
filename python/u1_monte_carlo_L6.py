"""
u1_monte_carlo_L6.py

Compact U(1) lattice gauge theory Monte Carlo simulation on a 4D periodic lattice.
This script performs thermalization and measurement of the plaquette observable
using the Metropolis algorithm with Numba JIT acceleration.

Part of UDCT v2.6.7: Higgs Hierarchy Stabilization via Adaptive Memory Kernel

Key features:
- Wilson plaquette action
- Periodic boundary conditions
- Metropolis update with staple summation
- Separate thermalization and measurement phases
- Data saved as .npy for later analysis with Memory Kernel

Author: Won Shik Paik
"""

import numpy as np
from numba import njit


@njit
def plaquette_angle(theta, x, y, z, t, mu, nu, L):
    """
    Calculate the plaquette angle at a given link position and directions.

    Parameters
    ----------
    theta : np.ndarray
        Link angle field of shape (L, L, L, L, 4)
    x, y, z, t : int
        Lattice coordinates
    mu, nu : int
        Directions of the two links forming the plaquette
    L : int
        Lattice size

    Returns
    -------
    float
        Plaquette angle (sum of oriented link angles around the square)
    """
    # Forward links
    p1 = theta[x, y, z, t, mu]
    p2 = theta[(x + (mu == 0)) % L, (y + (mu == 1)) % L,
               (z + (mu == 2)) % L, (t + (mu == 3)) % L, nu]
    # Backward links
    p3 = -theta[(x + (nu == 0)) % L, (y + (nu == 1)) % L,
                (z + (nu == 2)) % L, (t + (nu == 3)) % L, mu]
    p4 = -theta[x, y, z, t, nu]

    return p1 + p2 + p3 + p4


@njit
def calculate_deltaS(theta, x, y, z, t, mu, beta, L):
    """
    Calculate the change in action (Delta S) for a proposed link update.

    Uses the staple summation method for efficiency.
    """
    delta_S = 0.0
    for nu in range(4):
        if nu == mu:
            continue
        # Two staple directions
        staple = (plaquette_angle(theta, x, y, z, t, mu, nu, L) +
                  plaquette_angle(theta, x, y, z, t, nu, mu, L))
        delta_S += np.cos(staple)
    return -beta * delta_S


@njit
def metropolis_sweep(theta, beta, L, n_sweeps=1):
    """
    Perform one or more Metropolis sweeps over the entire lattice.
    """
    accepted = 0
    total = 0

    for _ in range(n_sweeps):
        for x in range(L):
            for y in range(L):
                for z in range(L):
                    for t in range(L):
                        for mu in range(4):
                            old_link = theta[x, y, z, t, mu]
                            # Propose new link angle
                            new_link = old_link + np.random.uniform(-0.3, 0.3)

                            # Calculate action change
                            delta_S = calculate_deltaS(theta, x
