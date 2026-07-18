"""
u1_monte_carlo_L6.py

Compact U(1) lattice gauge theory Monte Carlo simulation on L=6 lattice.
This script performs thermalization and measurement using the Metropolis algorithm
with Numba JIT compilation for performance.

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
    Calculate the plaquette angle at a given position and directions.
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
    Uses the staple summation method.
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
                            new_link = new_link % (2 * np.pi)

                            # Calculate action change
                            delta_S = calculate_deltaS(theta, x, y, z, t, mu, beta, L)

                            # Metropolis acceptance
                            if np.random.random() < np.exp(-delta_S):
                                theta[x, y, z, t, mu] = new_link
                                accepted += 1
                            total += 1
    return accepted / total if total > 0 else 0.0


@njit
def calculate_average_plaquette(theta, L):
    """
    Calculate the average plaquette value over the entire lattice.
    """
    total = 0.0
    count = 0
    for x in range(L):
        for y in range(L):
            for z in range(L):
                for t in range(L):
                    for mu in range(4):
                        for nu in range(mu + 1, 4):
                            angle = plaquette_angle(theta, x, y, z, t, mu, nu, L)
                            total += np.cos(angle)
                            count += 1
    return total / count


def main():
    """
    Main function to run U(1) Monte Carlo simulation on L=6 lattice.
    """
    print("=" * 60)
    print("UDCT v2.6.7 - U(1) Lattice Monte Carlo Simulation (L=6)")
    print("=" * 60)

    # === Parameters ===
    L = 6
    beta = 1.0
    n_thermalization = 600
    n_measurement = 3000
    sweeps_per_measurement = 5

    print(f"\nLattice size L = {L}")
    print(f"Beta = {beta}")
    print(f"Thermalization sweeps: {n_thermalization}")
    print(f"Measurement sweeps: {n_measurement}")
    print(f"Sweeps per measurement: {sweeps_per_measurement}")

    # === Initialize lattice ===
    print("\nInitializing random lattice...")
    theta = np.random.uniform(0, 2 * np.pi, size=(L, L, L, L, 4)).astype(np.float64)

    # === Thermalization ===
    print("\nStarting thermalization...")
    for i in range(n_thermalization):
        acc_rate = metropolis_sweep(theta, beta, L, n_sweeps=1)
        if (i + 1) % 100 == 0:
            print(f"  Thermalization step {i+1}/{n_thermalization} | Acceptance: {acc_rate:.3f}")

    print("Thermalization completed.")

    # === Measurement ===
    print("\nStarting measurement phase...")
    plaquette_series = []

    for i in range(n_measurement):
        metropolis_sweep(theta, beta, L, n_sweeps=sweeps_per_measurement)
        plaq = calculate_average_plaquette(theta, L)
        plaquette_series.append(plaq)

        if (i + 1) % 500 == 0:
            print(f"  Measurement {i+1}/{n_measurement} | Plaquette: {plaq:.6f}")

    plaquette_series = np.array(plaquette_series)

    # === Save results ===
    output_file = "plaquette_series_L6_beta1.0.npy"
    np.save(output_file, plaquette_series)

    print("\n" + "=" * 60)
    print("Simulation completed successfully!")
    print(f"Data saved as: {output_file}")
    print(f"Number of measurements: {len(plaquette_series)}")
    print(f"Mean plaquette: {np.mean(plaquette_series):.6f}")
    print(f"Std plaquette : {np.std(plaquette_series):.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
