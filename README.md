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
cd DIffUCO

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

Train on synthetic chip placement instances:

```bash
python argparse_ray_main.py \
    --EnergyFunction ChipPlacement \
    --IsingMode Chip_20_components \
    --train_mode PPO \
    --n_diffusion_steps 50 \
    --batch_size 16 \
    --n_basis_states 10 \
    --GPUs 0
```

### Configuration

Key hyperparameters can be set via command-line arguments or modified in `chip_placement_config.py`:

- `--n_diffusion_steps`: Number of diffusion timesteps (default: 50)
- `--batch_size`: Number of circuits per batch (default: 16)
- `--n_basis_states`: Parallel trajectories per circuit (default: 10)
- `--overlap_weight`: Weight for overlap penalty (default: 50.0)
- `--boundary_weight`: Weight for boundary penalty (default: 50.0)

### Dataset

The code generates synthetic chip placement instances with realistic characteristics:
- Component sizes: Clipped exponential distribution
- Netlist connectivity: Proximity-based k-NN graphs
- Circuit sizes: 20-100 components (configurable)

Available dataset configurations:
- `Chip_dummy`: 5 components (fast testing)
- `Chip_20_components`: 20-50 components (recommended)
- `Chip_50_components`: 50-100 components
- `Chip_100_components`: 100-200 components

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
- **Per-Step Energy**: Dense feedback at every diffusion timestep
- **GAE**: Generalized Advantage Estimation (λ=0.95, γ=0.99)
- **Learning Rate**: 3×10⁻⁴ with cosine annealing

## Results

Our method achieves competitive results on synthetic benchmarks:

| Dataset | Components | HPWL | Overlap | Runtime |
|---------|-----------|------|---------|---------|
| Chip_20 | 20-50 | 2.84×10⁷ | 0% | 50ms |
| Chip_50 | 50-100 | 4.21×10⁷ | 0% | 120ms |

*Detailed results will be included upon publication.*

## Project Structure

```
DIffUCO/
├── argparse_ray_main.py          # Main training entry point
├── train.py                      # Training loop
├── chip_placement_config.py      # Configuration presets
├── EnergyFunctions/
│   ├── ChipPlacementEnergy.py   # Energy function implementation
│   └── BaseEnergy.py            # Base energy class
├── Networks/
│   ├── DiffModel.py             # GNN diffusion model
│   └── Modules/                 # GNN components
├── Trainers/
│   ├── PPO_Trainer.py           # PPO training algorithm
│   └── BaseTrainer.py           # Base trainer class
├── NoiseDistributions/
│   └── GaussianNoise.py         # Continuous noise distribution
└── DatasetCreator/
    └── loadGraphDatasets/       # Synthetic data generation
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{anonymous2026energy,
  title={Energy-Guided Continuous Diffusion for Unsupervised Chip Placement},
  author={Anonymous},
  booktitle={Design Automation Conference (DAC)},
  year={2026}
}
```

## License

[To be determined upon publication]

## Acknowledgments

This work builds upon advances in diffusion models, reinforcement learning, and physical design automation.
