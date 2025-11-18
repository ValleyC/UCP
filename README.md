# Energy-Guided Continuous Diffusion for Chip Placement

Official implementation for the paper "Energy-Guided Continuous Diffusion for Unsupervised Chip Placement" (DAC 2026 submission).

## Overview

This repository contains the implementation of an energy-guided diffusion model for automated chip placement. The method combines:

- **Continuous diffusion processes** for 2D component positioning
- **Energy-guided policy gradient training** (PPO) with dense per-step feedback
- **Unsupervised learning** without requiring optimal placement examples
- **Smooth differentiable energy function** (HPWL + overlap + boundary penalties)

## Key Features

- **Unsupervised Training**: Learns directly from energy minimization without labeled data
- **Per-Step Energy Guidance**: Dense feedback at every diffusion timestep for efficient learning
- **Parallel Optimization**: Optimizes all components simultaneously using graph neural networks
- **Zero-Shot Generalization**: Generalizes to new circuit sizes without retraining

## Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- JAX 0.4.13+

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd UCP

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

Train on synthetic chip placement instances:

```bash
python argparse_ray_main.py \
    --EnergyFunction ChipPlacement \
    --dataset Chip_20_components \
    --train_mode PPO \
    --n_diffusion_steps 50 \
    --batch_size 16 \
    --n_basis_states 10 \
    --GPUs 0
```

### Configuration

Key hyperparameters can be set via command-line arguments or modified in `chip_placement_config.py`:

- `--n_diffusion_steps`: Number of diffusion timesteps (default: 1000)
- `--batch_size`: Number of circuits per batch (default: 32)
- `--n_basis_states`: Parallel trajectories per circuit (default: 20)
- `--overlap_weight`: Weight for overlap penalty (default: 2.0)
- `--boundary_weight`: Weight for boundary penalty (default: 1.0)
- `--lr`: Learning rate (default: 3e-4)

### Dataset

The code generates synthetic chip placement instances with realistic characteristics matching the paper specifications:

**Component Generation:**
- Circuit sizes: N ~ Uniform(200, 1000) components per instance
- Component sizes: Clipped exponential distribution with λ ∈ [0.04, 0.08]
- Size bounds: [s_min, s_max] ∈ [0.01, 1.0]
- Canvas aspect ratios: Uniform(0.5, 2.0)

**Netlist Connectivity (3 types):**
- **Local proximity (60%)**: Each component connects to k_local ~ Uniform{2, 4} nearest neighbors
- **Hierarchical clusters (30%)**: K-means clusters (m ~ Uniform(8, 16) per cluster) with weighted inter-cluster connections
- **Long-range (10%)**: Random component pairs simulating critical paths
- **Multi-pin nets**: 30% of 2-pin nets merged (60% → 3-pin, 30% → 4-pin, 10% → 5-pin)

**Training Data:**
- 20,000 synthetically generated circuits
- Random initial placements: (x, y) ~ Uniform(-1, 1)
- Ensures diversity for zero-shot generalization to real benchmarks

## Architecture

### Energy Function

The placement quality is evaluated using:

```
Energy = HPWL + λ_overlap × (overlap/τ_overlap)² + λ_boundary × (boundary/τ_boundary)²
```

Where:
- **HPWL**: Half-perimeter wirelength (sum of net bounding boxes)
- **Overlap**: Smooth pairwise component overlap penalty
- **Boundary**: Quadratic penalty for components exceeding canvas bounds
- Normalization thresholds prevent energy explosion

### Network

- **Architecture**: Graph Neural Network (Encode-Process-Decode)
- **Message Passing**: 5-8 layers with 64-128 hidden dimensions
- **Output**: Gaussian distribution parameters (mean, log-variance) for 2D positions
- **Input**: Component sizes, netlist connectivity, timestep encoding

### Training

- **Algorithm**: Proximal Policy Optimization (PPO)
- **Optimizer**: Adam (lr=3e-4, β₁=0.9, β₂=0.999, ε=1e-8)
- **PPO Settings**: ε_clip=0.2, 4 epochs per batch, minibatch size=8
- **Per-Step Energy**: Dense feedback at every diffusion timestep
- **GAE**: Generalized Advantage Estimation (λ=0.95, γ=0.99)
- **Value Loss Weight**: c_value=0.5
- **Gradient Clipping**: Max norm 0.5
- **Training**: 1000 epochs, batch size 32, 20,000 synthetic circuits
- **Diffusion**: 1000 steps, cosine schedule, β_t ∈ [0.0001, 0.02]
- **Reward Coefficients**: α_noise=0.01, α_ent=0.001
- **Energy Weights**: λ_overlap=2.0, λ_bound=1.0