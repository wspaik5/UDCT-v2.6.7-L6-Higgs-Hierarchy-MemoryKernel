# UDCT v2.6.7: Higgs Hierarchy Stabilization via Adaptive Memory Kernel

This repository contains the Python implementation and analysis codes for **UDCT v2.6.7**, focusing on an **Adaptive Memory Kernel** designed to improve the stability of Monte Carlo simulations in compact U(1) lattice gauge theory.

## Overview

This work implements an Adaptive Memory Kernel that combines three key components:

- Welford’s online algorithm for numerically stable mean and variance estimation
- Median-based outlier rejection to protect memory from large statistical fluctuations
- Adaptive exponential moving average with explicit warm-up phase and phase-jump detection

The kernel is applied to plaquette time series generated from L=6 U(1) lattice Monte Carlo simulations at β=1.0, demonstrating approximately 10% noise reduction while preserving the physical signal.

## Files

| File | Description |
|------|-------------|
| `u1_monte_carlo_L6.py` | Numba-accelerated U(1) lattice Monte Carlo simulator (L=6, β=1.0) |
| `memory_kernel.py` | Core Adaptive Memory Kernel implementation (Welford + Outlier + Phase-Jump) |
| `apply_memory_kernel_L6.py` | Applies the kernel to simulation data and generates comparison figures |
| `README.md` | This file |

## Requirements

```bash
pip install numpy numba matplotlib
