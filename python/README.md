# UDCT v2.6.7: Higgs Hierarchy Stabilization via Adaptive Memory Kernel

This repository contains the Python implementation and analysis codes for **UDCT v2.6.7**, focusing on an **Adaptive Memory Kernel** designed to improve the stability of Monte Carlo simulations in compact U(1) lattice gauge theory.

## Overview

The Adaptive Memory Kernel combines three key features:
- Welford’s online algorithm for numìerically stable mean and variance estimation
- Outlier rejection to protect memory from large statistical fluctuations
- Adaptive exponential moving average with warm-up phase and phase-jump detection

These features enable more stable long-running simulations, which is particularly useful for research on **Higgs hierarchy stabilization**.

## Key Results (L=6 Simulation)

- Lattice size: L=6, β=1.0
- Raw data: mean ≈ 0.6137, std ≈ 0.0279
- After Memory Kernel: mean ≈ 0.6117, std ≈ 0.0251
- **Noise reduction**: approximately **10%**
- The kernel effectively suppresses statistical outliers while preserving the physical signal.

## Repository Structure




## How to Run

1. Generate raw simulation data:
   ```bash
   python u1_monte_carlo_L6.py


python apply_memory_kernel_L6.py


Technical Note
A detailed technical note is available on Zenodo:
UDCT v2.6.7: Higgs Hierarchy Stabilization via Adaptive Memory Kernel – Python Implementation Technical Note
DOI: [To be added after Zenodo upload]
Citation
If you use this code or the results in your research, please cite:
Paik, W. S. (2026). UDCT v2.6.7: Higgs Hierarchy Stabilization via Adaptive Memory Kernel – Python Implementation Technical Note. Zenodo.
License
This project is licensed under the MIT License.
When using this code for research, please cite the accompanying technical note.
Related Works
[v2.6.6] Adaptive Threshold Memory Kernel Technical Note
[v2.6.5] Memory Kernel – Python Implementation Technical Note
[v2.6.4] Plaquette Susceptibility Technical Note

   
