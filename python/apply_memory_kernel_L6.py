"""
apply_memory_kernel_L6.py

Main script to apply the Adaptive Memory Kernel to L=6 U(1) lattice
Monte Carlo simulation results.

This script:
- Loads pre-computed plaquette time series from .npy file
- Applies the Memory Kernel with Welford statistics and outlier rejection
- Compares raw vs smoothed statistics (mean, std)
- Generates visualization (time series + histogram)
- Saves the resulting figure

Part of UDCT v2.6.7: Higgs Hierarchy Stabilization via Adaptive Memory Kernel

Author: Won Shik Paik
"""

import numpy as np
import matplotlib.pyplot as plt
from memory_kernel import memory_kernel


def main():
    """
    Main execution function for applying Memory Kernel to L=6 data.
    """
    # === Configuration ===
    input_file = "plaquette_series_L6_beta1.0.npy"
    output_figure = "memory_kernel_result_L6.png"

    alpha = 0.12
    outlier_threshold = 0.008
    large_diff_threshold = 0.003
    warm_up_steps = 20

    print("=" * 60)
    print("UDCT v2.6.7 - Applying Adaptive Memory Kernel (L=6)")
    print("=" * 60)

    # === Load simulation data ===
    try:
        raw_data = np.load(input_file)
        print(f"\nLoaded data from: {input_file}")
        print(f"Number of measurements: {len(raw_data)}")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        print("Please run u1_monte_carlo_L6.py first to generate the data.")
        return

    # === Apply Memory Kernel ===
    print("\nApplying Adaptive Memory Kernel...")
    smoothed_data = memory_kernel(
        raw_data,
        alpha=alpha,
        outlier_threshold=outlier_threshold,
        large_diff_threshold=large_diff_threshold,
        warm_up_steps=warm_up_steps
    )

    # === Statistical Comparison ===
    raw_mean = np.mean(raw_data)
    raw_std = np.std(raw_data)
    smoothed_mean = np.mean(smoothed_data)
    smoothed_std = np.std(smoothed_data)

    noise_reduction = (1 - smoothed_std / raw_std) * 100

    print("\n" + "-" * 50)
    print("Statistical Comparison: Raw vs Smoothed")
    print("-" * 50)
    print(f"Raw Mean      : {raw_mean:.6f}")
    print(f"Smoothed Mean : {smoothed_mean:.6f}")
    print(f"Raw Std       : {raw_std:.6f}")
    print(f"Smoothed Std  : {smoothed_std:.6f}")
    print(f"Noise Reduction: {noise_reduction:.2f}%")
    print("-" * 50)

    # === Visualization ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Time series comparison
    axes[0].plot(raw_data, label="Raw Plaquette", color="#1f77b4", alpha=0.6, linewidth=1)
    axes[0].plot(smoothed_data, label="After Memory Kernel", color="#d62728", linewidth=1.8)
    axes[0].set_xlabel("Monte Carlo Step", fontsize=12)
    axes[0].set_ylabel("Plaquette Value", fontsize=12)
    axes[0].set_title("Raw vs Memory Kernel Smoothed (L=6, β=1.0)", fontsize=13, fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Right: Histogram comparison
    axes[1].hist(raw_data, bins=40, alpha=0.5, label="Raw", color="#1f77b4", density=True)
    axes[1].hist(smoothed_data, bins=40, alpha=0.7, label="Smoothed", color="#d62728", density=True)
    axes[1].set_xlabel("Plaquette Value", fontsize=12)
    axes[1].set_ylabel("Density", fontsize=12)
    axes[1].set_title("Distribution Comparison", fontsize=13, fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_figure, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved as: {output_figure}")
    plt.show()

    print("\nMemory Kernel application completed successfully.")


if __name__ == "__main__":
    main()
