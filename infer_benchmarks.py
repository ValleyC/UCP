"""
UCP Benchmark Inference Pipeline

Runs UCP model on standard chip placement benchmarks (ISPD2005, IBM ICCAD04).

Usage:
    python infer_benchmarks.py \\
        --checkpoint logs/model.ckpt \\
        --benchmark_path ../datasets \\
        --dataset ispd2005 \\
        --output_dir results/ispd2005 \\
        --num_samples 20

Supported Datasets:
    - ispd2005: ISPD 2005 benchmarks (8 circuits)
    - clustered-ibm: IBM benchmarks with 512-cluster preprocessing
    - macro-ibm: IBM benchmarks, macro-only placement
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pickle
import time

# Add UCP to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_utils import (
    load_benchmark_dataset,
    save_placement,
    normalize_positions,
)


def compute_hpwl(positions, H_graph, normalized=True):
    """
    Compute Half-Perimeter Wirelength

    Args:
        positions: (V, 2) array of positions
        H_graph: jraph.GraphsTuple with netlist
        normalized: Whether to normalize by number of nets

    Returns:
        hpwl: Total wirelength
    """
    total_hpwl = 0.0
    num_nets = H_graph.edges.shape[0] // 2  # Bidirectional edges

    for i in range(num_nets):
        u = H_graph.senders[i]
        v = H_graph.receivers[i]

        # Get terminal positions
        u_pos = positions[u]
        v_pos = positions[v]

        # Add terminal offsets if available
        if H_graph.edges is not None:
            u_offset = H_graph.edges[i, :2]
            v_offset = H_graph.edges[i, 2:]
            u_pos = u_pos + u_offset
            v_pos = v_pos + v_offset

        # Bounding box half-perimeter
        bbox_width = abs(u_pos[0] - v_pos[0])
        bbox_height = abs(u_pos[1] - v_pos[1])
        total_hpwl += (bbox_width + bbox_height)

    if normalized:
        total_hpwl /= max(num_nets, 1)

    return total_hpwl


def compute_overlap(positions, sizes):
    """
    Compute total component overlap

    Args:
        positions: (V, 2) center positions
        sizes: (V, 2) component sizes

    Returns:
        overlap_ratio: Fraction of overlapping area
    """
    V = positions.shape[0]
    total_overlap = 0.0

    for i in range(V):
        for j in range(i + 1, V):
            # Get bounding boxes
            i_left = positions[i, 0] - sizes[i, 0] / 2
            i_right = positions[i, 0] + sizes[i, 0] / 2
            i_bottom = positions[i, 1] - sizes[i, 1] / 2
            i_top = positions[i, 1] + sizes[i, 1] / 2

            j_left = positions[j, 0] - sizes[j, 0] / 2
            j_right = positions[j, 0] + sizes[j, 0] / 2
            j_bottom = positions[j, 1] - sizes[j, 1] / 2
            j_top = positions[j, 1] + sizes[j, 1] / 2

            # Check overlap
            overlap_x = max(0, min(i_right, j_right) - max(i_left, j_left))
            overlap_y = max(0, min(i_top, j_top) - max(i_bottom, j_bottom))
            overlap_area = overlap_x * overlap_y

            total_overlap += overlap_area

    # Compute total area
    total_area = np.sum(sizes[:, 0] * sizes[:, 1])

    return total_overlap / max(total_area, 1e-10)


def compute_energy(positions, H_graph, metadata, overlap_weight=2.0, boundary_weight=1.0):
    """
    Compute placement energy (HPWL + penalties)

    Args:
        positions: (V, 2) positions
        H_graph: jraph.GraphsTuple
        metadata: dict with chip_size, etc.
        overlap_weight: Weight for overlap penalty
        boundary_weight: Weight for boundary penalty

    Returns:
        energy: Total energy
        metrics: Dict with individual components
    """
    sizes = H_graph.nodes

    # HPWL
    hpwl = compute_hpwl(positions, H_graph, normalized=False)

    # Overlap penalty
    overlap = compute_overlap(positions, sizes)

    # Boundary penalty
    chip_size = metadata.get('chip_size', (2.0, 2.0))
    boundary_violations = 0.0

    for i in range(positions.shape[0]):
        left = positions[i, 0] - sizes[i, 0] / 2
        right = positions[i, 0] + sizes[i, 0] / 2
        bottom = positions[i, 1] - sizes[i, 1] / 2
        top = positions[i, 1] + sizes[i, 1] / 2

        # Assuming chip is centered at origin with size chip_size
        x_min, x_max = -chip_size[0] / 2, chip_size[0] / 2
        y_min, y_max = -chip_size[1] / 2, chip_size[1] / 2

        # Boundary violations
        if left < x_min:
            boundary_violations += (x_min - left) ** 2
        if right > x_max:
            boundary_violations += (right - x_max) ** 2
        if bottom < y_min:
            boundary_violations += (y_min - bottom) ** 2
        if top > y_max:
            boundary_violations += (top - y_max) ** 2

    # Total energy
    energy = hpwl + overlap_weight * overlap + boundary_weight * boundary_violations

    metrics = {
        'hpwl': hpwl,
        'overlap': overlap,
        'boundary': boundary_violations,
        'energy': energy,
    }

    return energy, metrics


def run_ucp_inference(
    model,
    H_graph,
    metadata,
    num_samples=20,
    num_diffusion_steps=1000,
    seed=42,
):
    """
    Run UCP diffusion model inference

    Args:
        model: Trained UCP model
        H_graph: jraph.GraphsTuple
        metadata: Dict with chip_size, etc.
        num_samples: Number of samples to generate
        num_diffusion_steps: Diffusion timesteps
        seed: Random seed

    Returns:
        best_placement: (V, 2) best placement
        all_placements: List of all placements
        all_metrics: List of metric dicts
    """
    num_nodes = H_graph.nodes.shape[0]
    rng = jax.random.PRNGKey(seed)

    all_placements = []
    all_metrics = []

    for i in range(num_samples):
        rng, sample_rng = jax.random.split(rng)

        # Initialize random placement
        initial_positions = jax.random.uniform(
            sample_rng,
            shape=(num_nodes, 2),
            minval=-1.0,
            maxval=1.0
        )

        # Run UCP diffusion reverse sampling
        # TODO: Replace with actual UCP model call
        # placement = model.reverse_sample(
        #     H_graph,
        #     initial_state=initial_positions,
        #     num_steps=num_diffusion_steps,
        #     rng=sample_rng
        # )

        # Placeholder: Use random placement for now
        placement = np.array(initial_positions)

        # Compute metrics
        energy, metrics = compute_energy(
            placement,
            H_graph,
            metadata,
            overlap_weight=2.0,
            boundary_weight=1.0
        )

        all_placements.append(placement)
        all_metrics.append(metrics)

        print(f"  Sample {i+1}/{num_samples}: "
              f"HPWL={metrics['hpwl']:.2e}, "
              f"Overlap={metrics['overlap']:.4f}, "
              f"Energy={energy:.2e}")

    # Select best by energy
    best_idx = np.argmin([m['energy'] for m in all_metrics])
    best_placement = all_placements[best_idx]
    best_metrics = all_metrics[best_idx]

    print(f"\n  Best sample: {best_idx}")
    print(f"    HPWL: {best_metrics['hpwl']:.2e}")
    print(f"    Overlap: {best_metrics['overlap']:.4f}")
    print(f"    Energy: {best_metrics['energy']:.2e}")

    return best_placement, all_placements, all_metrics


def main():
    parser = argparse.ArgumentParser(description='UCP Benchmark Inference')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to UCP model checkpoint')
    parser.add_argument('--benchmark_path', type=str, required=True,
                        help='Path to benchmark datasets directory')
    parser.add_argument('--dataset', type=str, default='ispd2005',
                        help='Dataset name (ispd2005, clustered-ibm, macro-ibm)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for placements')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of placement samples per benchmark')
    parser.add_argument('--num_diffusion_steps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_all_samples', action='store_true',
                        help='Save all samples (not just best)')

    args = parser.parse_args()

    print("=" * 80)
    print("UCP Benchmark Inference Pipeline")
    print("=" * 80)
    print(f"Benchmark path: {args.benchmark_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"Samples per benchmark: {args.num_samples}")
    print(f"Diffusion steps: {args.num_diffusion_steps}")
    print("=" * 80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = None
    if args.checkpoint:
        print(f"\nLoading UCP model from {args.checkpoint}...")
        # TODO: Implement model loading
        # from train import load_checkpoint
        # model = load_checkpoint(args.checkpoint)
        print("WARNING: Model loading not implemented - using random placement")
    else:
        print("\nNo checkpoint provided - using random placement baseline")

    # Load benchmarks
    print(f"\nLoading {args.dataset} benchmarks...")
    benchmarks = load_benchmark_dataset(args.benchmark_path, args.dataset)
    print(f"Loaded {len(benchmarks)} benchmark circuits\n")

    # Results tracking
    all_results = []

    # Run inference on each benchmark
    for H_graph, metadata, bench_idx in benchmarks:
        print(f"\n{'=' * 70}")
        print(f"Benchmark {bench_idx}")
        print(f"  Components: {H_graph.nodes.shape[0]}")
        print(f"  Nets: {H_graph.edges.shape[0] // 2}")
        print(f"  Canvas size: {metadata['chip_size']}")
        print(f"{'=' * 70}")

        start_time = time.time()

        # Run UCP inference
        best_placement, all_placements, all_metrics = run_ucp_inference(
            model,
            H_graph,
            metadata,
            num_samples=args.num_samples,
            num_diffusion_steps=args.num_diffusion_steps,
            seed=args.seed + bench_idx,
        )

        runtime = time.time() - start_time
        print(f"\n  Runtime: {runtime:.2f} seconds")

        # Save best placement
        best_output_path = output_dir / f"sample{bench_idx}.pkl"
        save_placement(best_placement, best_output_path)

        # Optionally save all samples
        if args.save_all_samples:
            samples_dir = output_dir / "samples"
            samples_dir.mkdir(exist_ok=True)
            for i, placement in enumerate(all_placements):
                sample_path = samples_dir / f"bench{bench_idx}_sample{i}.pkl"
                save_placement(placement, sample_path)

        # Track results
        best_metrics = all_metrics[np.argmin([m['energy'] for m in all_metrics])]
        all_results.append({
            'benchmark': bench_idx,
            'hpwl': best_metrics['hpwl'],
            'overlap': best_metrics['overlap'],
            'energy': best_metrics['energy'],
            'runtime': runtime,
        })

    # Save summary
    summary_path = output_dir / "results_summary.pkl"
    with open(summary_path, 'wb') as f:
        pickle.dump(all_results, f)

    # Print summary
    print(f"\n{'=' * 80}")
    print("Results Summary")
    print(f"{'=' * 80}")
    print(f"{'Benchmark':<12} {'HPWL':>12} {'Overlap':>10} {'Energy':>12} {'Time (s)':>10}")
    print("-" * 80)

    for result in all_results:
        print(f"{result['benchmark']:<12} "
              f"{result['hpwl']:>12.2e} "
              f"{result['overlap']:>10.4f} "
              f"{result['energy']:>12.2e} "
              f"{result['runtime']:>10.2f}")

    # Compute averages
    avg_hpwl = np.mean([r['hpwl'] for r in all_results])
    avg_overlap = np.mean([r['overlap'] for r in all_results])
    avg_energy = np.mean([r['energy'] for r in all_results])
    avg_time = np.mean([r['runtime'] for r in all_results])

    print("-" * 80)
    print(f"{'Average':<12} "
          f"{avg_hpwl:>12.2e} "
          f"{avg_overlap:>10.4f} "
          f"{avg_energy:>12.2e} "
          f"{avg_time:>10.2f}")
    print("=" * 80)

    print(f"\nAll results saved to {output_dir}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
