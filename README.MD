# UDCT v2.6.7: L=6 Monte Carlo + Adaptive Memory Kernel for Higgs Hierarchy Stabilization

This repository contains the Python implementation and results for **UDCT v2.6.7**, focusing on applying an **Adaptive Memory Kernel** to compact U(1) lattice gauge theory simulations on a small lattice (L=6).

## Overview

- Monte Carlo simulation of compact U(1) lattice gauge theory at β=1.0 on a 6⁴ lattice
- Implementation of an improved Adaptive Memory Kernel with Welford’s online algorithm, outlier rejection, and phase jump detection
- Demonstration of noise reduction while preserving physical signals

## Files

### Python Code (`python/`)

| File | Description |
|------|-------------|
| `u1_monte_carlo_L6.py` | L=6 Monte Carlo simulation (thermalization + measurement) |
| `memory_kernel.py` | Adaptive Memory Kernel class with dynamic thresholding |
| `apply_memory_kernel_L6.py` | Applies Memory Kernel to simulation data and generates comparison plots |

### Figures (`figures/`)

| File | Description |
|------|-------------|
| `L6_memory_kernel_comparison.png` | Time series and distribution comparison (Raw vs Memory Kernel) |
| `L6_raw_plaquette.png` | Raw plaquette time series and distribution from Monte Carlo |

## How to Run

```bash
cd python
python u1_monte_carlo_L6.py          # Run Monte Carlo simulation
python apply_memory_kernel_L6.py     # Apply Memory Kernel and generate plots


If you use this code or results in your research, please cite the corresponding 
technical note:

UDCT v2.6.7: Higgs Hierarchy Stabilization via Adaptive Memory Kernel 
– Python Implementation Technical Note (2026)
