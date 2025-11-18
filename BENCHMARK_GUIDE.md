# UCP Benchmark Inference Guide

Complete guide for running UCP on standard chip placement benchmarks (ISPD2005, IBM ICCAD04).

## Overview

UCP provides a complete pipeline for:
1. Parsing raw benchmark files (DEF format)
2. Preprocessing (clustering, macro extraction)
3. Running inference with trained UCP models
4. Evaluating placement quality

## Quick Start

```bash
# 1. Run inference on preprocessed benchmarks
python infer_benchmarks.py \
    --checkpoint logs/ucp_model.ckpt \
    --benchmark_path ../datasets \
    --dataset ispd2005 \
    --output_dir results/ispd2005 \
    --num_samples 20

# 2. Parse raw benchmarks (if needed)
python parse_benchmarks.py \
    --input_dir benchmarks/ispd2005_raw \
    --output_dir datasets/graph/ispd2005 \
    --mode clustered \
    --num_clusters 512
```

## File Structure

```
UCP/
├── benchmark_utils.py      # Benchmark data utilities
├── infer_benchmarks.py     # Main inference script
├── parse_benchmarks.py     # Raw benchmark parser
└── BENCHMARK_GUIDE.md      # This file
```

## Data Formats

### Input Format (graph*.pickle)

Preprocessed benchmarks are stored as `torch_geometric.data.Data` objects:

```python
{
    'x': (V, 2) tensor,           # Component sizes [width, height]
    'edge_index': (2, E) tensor,  # Graph connectivity (bidirectional)
    'edge_attr': (E, 4) tensor,   # Terminal offsets [u_x, u_y, v_x, v_y]
    'is_ports': (V,) bool,        # Port mask
    'is_macros': (V,) bool,       # Macro mask
    'chip_size': tuple,           # Canvas size
}
```

### Output Format (sample*.pkl)

Placements are saved as numpy arrays:

```python
positions = np.array([[x0, y0],
                      [x1, y1],
                      ...,
                      [xN, yN]])  # Shape: (V, 2)
```

Coordinate system: Normalized to `[-1, 1]` for both x and y axes.

## Usage

### 1. Parsing Raw Benchmarks

Convert DEF files to preprocessed graph format:

```bash
# Clustered mode (512 clusters via hMetis)
python parse_benchmarks.py \
    --input_dir benchmarks/ispd2005_raw \
    --output_dir datasets/graph/ispd2005-clustered \
    --mode clustered \
    --num_clusters 512

# Macro-only mode
python parse_benchmarks.py \
    --input_dir benchmarks/ibm_raw \
    --output_dir datasets/graph/ibm-macro \
    --mode macro-only

# Raw mode (no preprocessing)
python parse_benchmarks.py \
    --input_dir benchmarks/ispd2005_raw \
    --output_dir datasets/graph/ispd2005-raw \
    --mode raw
```

**Requirements:**
- hMetis binary (`shmetis`) in PATH or current directory
- Download from: http://glaros.dtc.umn.edu/gkhome/metis/hmetis/overview

### 2. Loading Benchmarks

```python
from benchmark_utils import load_benchmark_dataset

# Load preprocessed benchmarks
benchmarks = load_benchmark_dataset(
    "datasets",
    dataset_name="ispd2005"
)

# Each benchmark is (H_graph, metadata, file_idx)
for H_graph, metadata, idx in benchmarks:
    print(f"Benchmark {idx}:")
    print(f"  Components: {H_graph.nodes.shape[0]}")
    print(f"  Nets: {H_graph.edges.shape[0] // 2}")
    print(f"  Canvas: {metadata['chip_size']}")
```

### 3. Running Inference

```bash
python infer_benchmarks.py \
    --checkpoint logs/model.ckpt \           # UCP model checkpoint
    --benchmark_path ../datasets \           # Path to datasets directory
    --dataset ispd2005 \                     # Dataset name
    --output_dir results/ispd2005 \          # Output directory
    --num_samples 20 \                       # Samples per benchmark
    --num_diffusion_steps 1000 \             # Diffusion timesteps
    --seed 42                                # Random seed
```

**Output:**
- `results/ispd2005/sample0.pkl` - Best placement for benchmark 0
- `results/ispd2005/sample1.pkl` - Best placement for benchmark 1
- ...
- `results/ispd2005/results_summary.pkl` - Summary statistics

### 4. Saving Placements

```python
from benchmark_utils import save_placement
import numpy as np

positions = np.array([[x0, y0], ..., [xN, yN]])
save_placement(positions, "results/sample0.pkl")
```

### 5. Computing Metrics

```python
from infer_benchmarks import compute_hpwl, compute_overlap, compute_energy

# HPWL
hpwl = compute_hpwl(positions, H_graph, normalized=False)

# Overlap ratio
overlap = compute_overlap(positions, sizes)

# Total energy
energy, metrics = compute_energy(
    positions,
    H_graph,
    metadata,
    overlap_weight=2.0,
    boundary_weight=1.0
)

print(f"HPWL: {metrics['hpwl']:.2e}")
print(f"Overlap: {metrics['overlap']:.4f}")
print(f"Energy: {energy:.2e}")
```

## Benchmark Datasets

### ISPD2005

**Description:** ISPD 2005 Placement Contest benchmarks

**Circuits:**
- adaptec1, adaptec2, adaptec3, adaptec4
- bigblue1, bigblue2, bigblue3, bigblue4

**Preprocessing:** 512-cluster aggregation using hMetis

**Usage:**
```bash
python infer_benchmarks.py \
    --benchmark_path ../datasets \
    --dataset ispd2005 \
    --output_dir results/ispd2005
```

### IBM ICCAD04

**Description:** IBM ICCAD 2004 placement benchmarks

**Modes:**
- **Clustered**: 512-cluster preprocessing
- **Macro-only**: Only macro components
- **Mixed**: Standard cells + macros

**Usage:**
```bash
# Clustered
python infer_benchmarks.py \
    --dataset clustered-ibm \
    --output_dir results/ibm-clustered

# Macro-only
python infer_benchmarks.py \
    --dataset macro-ibm \
    --output_dir results/ibm-macro
```

## Integration with UCP Model

To integrate with your trained UCP model, update `infer_benchmarks.py`:

```python
def run_ucp_inference(model, H_graph, metadata, ...):
    # Replace placeholder with actual UCP model call

    # Option 1: Using reverse diffusion
    placement = model.reverse_sample(
        graph=H_graph,
        num_steps=num_diffusion_steps,
        rng=sample_rng
    )

    # Option 2: Using policy network
    placement = model.generate_placement(
        graph=H_graph,
        metadata=metadata,
        num_diffusion_steps=num_diffusion_steps
    )

    return placement
```

## Evaluation Metrics

UCP computes standard placement metrics:

| Metric | Description | Formula |
|--------|-------------|---------|
| **HPWL** | Half-Perimeter Wirelength | Σ (bbox_width + bbox_height) for all nets |
| **Overlap** | Component overlap ratio | Σ overlap_area / Σ total_area |
| **Boundary** | Out-of-bounds penalty | Σ (distance_to_boundary)² |
| **Energy** | Total placement quality | HPWL + λ_overlap × overlap + λ_bound × boundary |

**Default Weights** (matching manuscript):
- λ_overlap = 2.0
- λ_boundary = 1.0

## Coordinate System

**Normalized Space:**
- x-axis: [-1, 1]
- y-axis: [-1, 1]
- Origin: Center of canvas (0, 0)

**Component Representation:**
- Position: (x, y) = center of component
- Size: (width, height)
- Bounding box: [x - w/2, x + w/2] × [y - h/2, y + h/2]

## Performance Optimization

### Multi-Sampling

Generate multiple samples and select best:

```bash
python infer_benchmarks.py \
    --num_samples 20 \
    --save_all_samples  # Optional: save all samples
```

### Batch Processing

Process multiple benchmarks in parallel:

```python
# TODO: Add parallel inference support
```

## Troubleshooting

### Missing hMetis

```
Error: shmetis not found
```

**Solution:** Download hMetis from http://glaros.dtc.umn.edu/gkhome/metis/hmetis/overview

### Empty Graph Files

```
Error: No graph*.pickle files found
```

**Solution:** Run `parse_benchmarks.py` first to preprocess raw DEF files

### Model Loading Error

```
WARNING: Model loading not implemented
```

**Solution:** Implement checkpoint loading in `infer_benchmarks.py`:
```python
from train import load_checkpoint
model = load_checkpoint(args.checkpoint)
```

## Complete Example

```python
from benchmark_utils import load_benchmark_dataset, save_placement
from infer_benchmarks import run_ucp_inference
import numpy as np

# 1. Load benchmarks
benchmarks = load_benchmark_dataset("../datasets", "ispd2005")

# 2. Get first benchmark
H_graph, metadata, idx = benchmarks[0]

# 3. Run UCP (placeholder - replace with actual model)
positions = np.random.uniform(-1, 1, size=(H_graph.nodes.shape[0], 2))

# 4. Compute metrics
from infer_benchmarks import compute_energy
energy, metrics = compute_energy(positions, H_graph, metadata)

print(f"HPWL: {metrics['hpwl']:.2e}")
print(f"Overlap: {metrics['overlap']:.4f}")

# 5. Save result
save_placement(positions, f"results/sample{idx}.pkl")
```

## Next Steps

1. **Train UCP Model:**
   ```bash
   python argparse_ray_main.py \
       --EnergyFunction ChipPlacement \
       --dataset Chip_20_components \
       --train_mode PPO \
       --n_diffusion_steps 1000 \
       --batch_size 32
   ```

2. **Run Inference:**
   ```bash
   python infer_benchmarks.py \
       --checkpoint logs/model.ckpt \
       --dataset ispd2005 \
       --output_dir results
   ```

3. **Analyze Results:**
   ```python
   import pickle
   with open('results/results_summary.pkl', 'rb') as f:
       results = pickle.load(f)
   ```

## References

- ISPD 2005: http://www.ispd.cc/contests/05/
- IBM Benchmarks: http://vlsicad.eecs.umich.edu/BK/Slots/cache/

www.public.iastate.edu/~nataraj/

- hMetis: http://glaros.dtc.umn.edu/gkhome/metis/hmetis/overview
