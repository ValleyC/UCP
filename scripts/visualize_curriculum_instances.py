#!/usr/bin/env python
"""
Visualize curriculum dataset instances to examine quality

Usage:
    python scripts/visualize_curriculum_instances.py --dataset Chip_v1_curriculum_stage1_n100 --n_instances 5
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import pickle

parser = argparse.ArgumentParser(description='Visualize curriculum dataset instances')
parser.add_argument('--dataset', type=str, required=True,
                   help='Dataset name (e.g., Chip_v1_curriculum_stage1_n100)')
parser.add_argument('--n_instances', type=int, default=5,
                   help='Number of instances to visualize')
parser.add_argument('--mode', type=str, default='train',
                   choices=['train', 'val', 'test'],
                   help='Which split to visualize')
parser.add_argument('--seed', type=int, default=123,
                   help='Random seed used during generation')
parser.add_argument('--save', action='store_true',
                   help='Save figures instead of showing them')
args = parser.parse_args()


def load_instance(dataset_name, mode, seed, idx):
    """Load a single instance from dataset"""
    data_dir = Path('DatasetCreator/loadGraphDatasets/DatasetSolutions/no_norm')
    instance_file = data_dir / dataset_name / mode / str(seed) / 'ChipPlacement' / 'indexed' / f'idx_{idx}_solutions.pickle'

    if not instance_file.exists():
        return None

    with open(instance_file, 'rb') as f:
        instance = pickle.load(f)

    return instance


def compute_overlap_area(pos1, size1, pos2, size2):
    """Compute overlap area between two rectangles"""
    x1_min, y1_min = pos1[0] - size1[0]/2, pos1[1] - size1[1]/2
    x1_max, y1_max = pos1[0] + size1[0]/2, pos1[1] + size1[1]/2

    x2_min, y2_min = pos2[0] - size2[0]/2, pos2[1] - size2[1]/2
    x2_max, y2_max = pos2[0] + size2[0]/2, pos2[1] + size2[1]/2

    dx = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    dy = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

    return dx * dy


def compute_boundary_violation(pos, size, canvas_bounds=(-1, 1)):
    """Compute boundary violation area"""
    x_min, y_min = pos[0] - size[0]/2, pos[1] - size[1]/2
    x_max, y_max = pos[0] + size[0]/2, pos[1] + size[1]/2

    violation = 0.0

    # Left boundary
    if x_min < canvas_bounds[0]:
        violation += (canvas_bounds[0] - x_min) * size[1]

    # Right boundary
    if x_max > canvas_bounds[1]:
        violation += (x_max - canvas_bounds[1]) * size[1]

    # Bottom boundary
    if y_min < canvas_bounds[0]:
        violation += (canvas_bounds[0] - y_min) * size[0]

    # Top boundary
    if y_max > canvas_bounds[1]:
        violation += (y_max - canvas_bounds[1]) * size[0]

    return violation


def compute_metrics(instance):
    """Compute placement metrics"""
    # Extract data
    legal_positions = instance['legal_positions']
    randomized_positions = instance['positions']
    sizes = instance['sizes']
    H_graph = instance['H_graphs']

    n_components = len(sizes)

    # Compute metrics for LEGAL placement
    total_overlap_legal = 0.0
    total_boundary_legal = 0.0

    for i in range(n_components):
        # Boundary violations
        total_boundary_legal += compute_boundary_violation(legal_positions[i], sizes[i])

        # Overlaps
        for j in range(i+1, n_components):
            total_overlap_legal += compute_overlap_area(
                legal_positions[i], sizes[i],
                legal_positions[j], sizes[j]
            )

    # Compute metrics for RANDOMIZED placement
    total_overlap_random = 0.0
    total_boundary_random = 0.0

    for i in range(n_components):
        total_boundary_random += compute_boundary_violation(randomized_positions[i], sizes[i])

        for j in range(i+1, n_components):
            total_overlap_random += compute_overlap_area(
                randomized_positions[i], sizes[i],
                randomized_positions[j], sizes[j]
            )

    # Compute density
    total_area = sum([size[0] * size[1] for size in sizes])
    canvas_area = 4.0
    density = total_area / canvas_area

    return {
        'n_components': n_components,
        'density': density,
        'overlap_legal': total_overlap_legal,
        'boundary_legal': total_boundary_legal,
        'overlap_random': total_overlap_random,
        'boundary_random': total_boundary_random,
        'hpwl': instance.get('Energies', 0.0)
    }


def visualize_instance(instance, idx, metrics):
    """Visualize a single instance with both legal and randomized placements"""

    legal_positions = instance['legal_positions']
    randomized_positions = instance['positions']
    sizes = instance['sizes']
    H_graph = instance['H_graphs']

    # Extract edges
    senders = H_graph.senders
    receivers = H_graph.receivers
    edge_attrs = H_graph.edges

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # === LEFT: Legal Placement ===
    ax = axes[0]
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(f'Legal Placement (Ground Truth)\nN={metrics["n_components"]}, '
                 f'Density={metrics["density"]:.3f}', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)

    # Draw canvas boundary
    canvas_rect = Rectangle((-1, -1), 2, 2, linewidth=2, edgecolor='black',
                           facecolor='none', linestyle='--', label='Canvas')
    ax.add_patch(canvas_rect)

    # Draw components
    for i, (pos, size) in enumerate(zip(legal_positions, sizes)):
        rect = Rectangle(
            (pos[0] - size[0]/2, pos[1] - size[1]/2),
            size[0], size[1],
            linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.6
        )
        ax.add_patch(rect)

        # Label component
        ax.text(pos[0], pos[1], str(i), ha='center', va='center',
               fontsize=6, color='darkblue')

    # Draw nets (sample a few to avoid clutter)
    n_edges_to_draw = min(50, len(senders))
    edge_indices = np.random.choice(len(senders), n_edges_to_draw, replace=False)

    for edge_idx in edge_indices:
        src = senders[edge_idx]
        dst = receivers[edge_idx]

        # Get terminal offsets
        src_offset = edge_attrs[edge_idx, :2]
        dst_offset = edge_attrs[edge_idx, 2:4]

        # Compute terminal positions
        src_pos = legal_positions[src] + src_offset
        dst_pos = legal_positions[dst] + dst_offset

        ax.plot([src_pos[0], dst_pos[0]], [src_pos[1], dst_pos[1]],
               'r-', alpha=0.3, linewidth=0.5)

    # Add metrics text
    metrics_text = (f"Overlap: {metrics['overlap_legal']:.4f}\n"
                   f"Boundary: {metrics['boundary_legal']:.4f}\n"
                   f"HPWL: {metrics['hpwl']:.1f}")
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=9)

    # === RIGHT: Randomized Placement ===
    ax = axes[1]
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(f'Randomized Placement (Training Input)\nN={metrics["n_components"]}, '
                 f'Density={metrics["density"]:.3f}', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)

    # Draw canvas boundary
    canvas_rect = Rectangle((-1, -1), 2, 2, linewidth=2, edgecolor='black',
                           facecolor='none', linestyle='--', label='Canvas')
    ax.add_patch(canvas_rect)

    # Draw components
    for i, (pos, size) in enumerate(zip(randomized_positions, sizes)):
        # Check if component violates boundaries
        x_min, y_min = pos[0] - size[0]/2, pos[1] - size[1]/2
        x_max, y_max = pos[0] + size[0]/2, pos[1] + size[1]/2
        violates_boundary = (x_min < -1 or x_max > 1 or y_min < -1 or y_max > 1)

        color = 'lightcoral' if violates_boundary else 'lightgreen'
        edge_color = 'red' if violates_boundary else 'green'

        rect = Rectangle(
            (pos[0] - size[0]/2, pos[1] - size[1]/2),
            size[0], size[1],
            linewidth=1, edgecolor=edge_color, facecolor=color, alpha=0.6
        )
        ax.add_patch(rect)

        # Label component
        ax.text(pos[0], pos[1], str(i), ha='center', va='center',
               fontsize=6, color='darkred' if violates_boundary else 'darkgreen')

    # Draw nets (same edges as left)
    for edge_idx in edge_indices:
        src = senders[edge_idx]
        dst = receivers[edge_idx]

        src_offset = edge_attrs[edge_idx, :2]
        dst_offset = edge_attrs[edge_idx, 2:4]

        src_pos = randomized_positions[src] + src_offset
        dst_pos = randomized_positions[dst] + dst_offset

        ax.plot([src_pos[0], dst_pos[0]], [src_pos[1], dst_pos[1]],
               'r-', alpha=0.3, linewidth=0.5)

    # Add metrics text
    metrics_text = (f"Overlap: {metrics['overlap_random']:.4f}\n"
                   f"Boundary: {metrics['boundary_random']:.4f}\n"
                   f"Note: High violations expected\n(model learns to fix these)")
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
           fontsize=9)

    plt.tight_layout()
    return fig


def main():
    print(f"\nVisualizing {args.n_instances} instances from {args.dataset} ({args.mode} split)\n")

    # Create output directory
    if args.save:
        output_dir = Path('figures') / 'curriculum_visualizations'
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving figures to {output_dir}/\n")

    # Load and visualize instances
    for i in range(args.n_instances):
        print(f"Loading instance {i}...")
        instance = load_instance(args.dataset, args.mode, args.seed, i)

        if instance is None:
            print(f"  Error: Instance {i} not found")
            continue

        # Compute metrics
        metrics = compute_metrics(instance)

        print(f"  N={metrics['n_components']}, Density={metrics['density']:.3f}, "
              f"Legal overlap={metrics['overlap_legal']:.4f}, "
              f"Random overlap={metrics['overlap_random']:.4f}")

        # Visualize
        fig = visualize_instance(instance, i, metrics)

        if args.save:
            filename = output_dir / f'{args.dataset}_instance_{i}.png'
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  Saved to {filename}")
            plt.close(fig)
        else:
            plt.show()

    if args.save:
        print(f"\n✓ All figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
