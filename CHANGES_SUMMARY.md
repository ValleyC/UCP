# UCP Repository Updates Summary

## Changes Made to Match DAC 2026 Manuscript

### 1. Repository Name Changes
- Changed all references from `DIffUCO` to `UCP` (Unsupervised Chip Placement)
- Updated README.md project structure
- Updated installation instructions

### 2. Argument Renaming
- Renamed `--IsingMode` to `--dataset` throughout codebase:
  - argparse_ray_main.py
  - chip_placement_config.py
  - generate_chip_dataset.py
  - generate_chip_curriculum.py

### 3. Hyperparameter Updates (chip_placement_config.py)
Updated to match manuscript specifications:
- Learning rate: `1e-4` → `3e-4`
- Batch size: `16` → `32`
- N_basis_states: `10` → `20`
- Diffusion steps: `50` → `1000`
- Diffusion schedule: `linear` → `cosine`
- PPO epochs: `2` → `4`
- Minibatch size: `5` → `8`
- Value loss weight: `0.65` → `0.5`

### 4. Energy Function Weights
Updated penalty weights to match manuscript:
- Overlap weight: `50.0` → `2.0` (λ_overlap)
- Boundary weight: `50.0` → `1.0` (λ_bound)

### 5. README.md Updates
- Updated dataset description to match manuscript specifications
- Added detailed training hyperparameters
- Updated component generation details:
  - Circuit sizes: N ~ Uniform(200, 1000)
  - Component sizes: Clipped exponential λ ∈ [0.04, 0.08]
  - Three-type connectivity: Local (60%), Hierarchical (30%), Long-range (10%)
  - Multi-pin net merging: 30% of nets
- Updated training details with full specifications

## Still Needed (Implementation)

### Dataset Generation Code (ChipDatasetGenerator_Unsupervised.py)
The netlist generation needs to be updated to implement:
1. **Component Generation**: N ~ Uniform(200, 1000) instead of current small values
2. **Three-Type Connectivity**:
   - Local proximity (60%): k_local ~ Uniform{2, 4} nearest neighbors
   - Hierarchical clusters (30%): k-means with m ~ Uniform(8, 16) per cluster
   - Long-range (10%): random component pairs
3. **Multi-pin Net Merging**: 30% of 2-pin nets merged (60%→3-pin, 30%→4-pin, 10%→5-pin)

Current implementation only uses simple k-NN (k=5), which doesn't match the manuscript's more sophisticated approach.

## Usage After Changes

Training command (updated):
```bash
python argparse_ray_main.py \
    --EnergyFunction ChipPlacement \
    --dataset Chip_20_components \
    --train_mode PPO \
    --n_diffusion_steps 1000 \
    --batch_size 32 \
    --n_basis_states 20 \
    --GPUs 0
```

Dataset generation command (updated):
```bash
python generate_chip_dataset.py --dataset Chip_medium --seed 123
```
