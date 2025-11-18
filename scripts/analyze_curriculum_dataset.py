#!/usr/bin/env python
"""
Analyze curriculum dataset to verify component count distribution

Usage:
    python scripts/analyze_curriculum_dataset.py --dataset Chip_v1_curriculum_stage1_n100
    python scripts/analyze_curriculum_dataset.py --all_stages
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

parser = argparse.ArgumentParser(description='Analyze curriculum dataset')
parser.add_argument('--dataset', type=str, default=None,
                   help='Dataset name (e.g., Chip_v1_curriculum_stage1_n100)')
parser.add_argument('--all_stages', action='store_true',
                   help='Analyze all curriculum stages')
parser.add_argument('--mode', type=str, default='train',
                   choices=['train', 'val', 'test'],
                   help='Which split to analyze')
parser.add_argument('--seed', type=int, default=123,
                   help='Random seed used during dataset generation')
args = parser.parse_args()

CURRICULUM_STAGES = [
    'Chip_v1_curriculum_stage1_n100',
    'Chip_v1_curriculum_stage2_n150',
    'Chip_v1_curriculum_stage3_n200',
    'Chip_v1_curriculum_stage4_n250',
    'Chip_v1_curriculum_stage5_n350'
]


def analyze_dataset(dataset_name, mode='train', seed=123):
    """Analyze component count distribution in a dataset"""

    # Load dataset - using the actual directory structure
    # Path: DatasetCreator/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{mode}/{seed}/ChipPlacement/indexed/
    data_dir = Path('DatasetCreator/loadGraphDatasets/DatasetSolutions/no_norm') / dataset_name / mode / str(seed) / 'ChipPlacement' / 'indexed'

    if not data_dir.exists():
        print(f"Error: Dataset not found at {data_dir}")
        print(f"\nTip: Make sure you're running this from the DIffUCO root directory")
        return None

    # Read all instances
    component_counts = []
    densities = []
    hpwls = []

    instance_files = sorted(data_dir.glob('idx_*_solutions.pickle'))

    if len(instance_files) == 0:
        print(f"Error: No instances found in {data_dir}")
        return None

    for instance_file in instance_files:
        with open(instance_file, 'rb') as f:
            instance = pickle.load(f)

        # Get number of components
        if 'graph_sizes' in instance:
            n_components = instance['graph_sizes']
        elif 'H_graphs' in instance:
            n_components = instance['H_graphs'].n_node[0]
        else:
            print(f"Warning: Could not determine component count for {instance_file}")
            continue

        component_counts.append(n_components)

        if 'densities' in instance:
            densities.append(instance['densities'])
        if 'Energies' in instance:
            hpwls.append(instance['Energies'])

    component_counts = np.array(component_counts)
    densities = np.array(densities) if densities else None
    hpwls = np.array(hpwls) if hpwls else None

    # Statistics
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} ({mode} split)")
    print(f"{'='*60}")
    print(f"Total instances: {len(component_counts)}")
    print(f"\nComponent Count Distribution:")
    print(f"  Min:     {component_counts.min()}")
    print(f"  Max:     {component_counts.max()}")
    print(f"  Mean:    {component_counts.mean():.1f}")
    print(f"  Median:  {np.median(component_counts):.1f}")
    print(f"  Std:     {component_counts.std():.1f}")

    # Percentiles
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  {p}%:     {np.percentile(component_counts, p):.1f}")

    if densities is not None:
        print(f"\nDensity Distribution:")
        print(f"  Mean:    {densities.mean():.3f}")
        print(f"  Range:   [{densities.min():.3f}, {densities.max():.3f}]")

    if hpwls is not None:
        print(f"\nHPWL Distribution (randomized placement):")
        print(f"  Mean:    {hpwls.mean():.1f}")
        print(f"  Range:   [{hpwls.min():.1f}, {hpwls.max():.1f}]")

    print(f"{'='*60}\n")

    return {
        'dataset_name': dataset_name,
        'component_counts': component_counts,
        'densities': densities,
        'hpwls': hpwls
    }


def plot_distributions(results):
    """Plot component count distributions for multiple datasets"""

    n_datasets = len(results)
    fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 4))

    if n_datasets == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        component_counts = result['component_counts']
        dataset_name = result['dataset_name']

        # Extract target from name (e.g., "stage1_n100" -> 100)
        try:
            target = int(dataset_name.split('_n')[-1])
            title = f"Stage: N={target}"
        except:
            title = dataset_name

        ax.hist(component_counts, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(component_counts.mean(), color='r', linestyle='--',
                   label=f'Mean: {component_counts.mean():.1f}')
        ax.axvline(np.median(component_counts), color='g', linestyle='--',
                   label=f'Median: {np.median(component_counts):.1f}')

        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save figure
    os.makedirs('figures', exist_ok=True)
    if len(results) == 1:
        filename = f"figures/curriculum_{results[0]['dataset_name']}_distribution.png"
    else:
        filename = "figures/curriculum_all_stages_distribution.png"

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {filename}")
    plt.show()


def main():
    results = []

    if args.all_stages:
        # Analyze all curriculum stages
        print("\n" + "="*70)
        print("ANALYZING ALL CURRICULUM STAGES")
        print("="*70)

        for dataset_name in CURRICULUM_STAGES:
            result = analyze_dataset(dataset_name, args.mode, args.seed)
            if result is not None:
                results.append(result)

        if results:
            plot_distributions(results)

    elif args.dataset:
        # Analyze specific dataset
        result = analyze_dataset(args.dataset, args.mode, args.seed)
        if result is not None:
            plot_distributions([result])

    else:
        print("Error: Must specify either --dataset or --all_stages")
        print("\nExamples:")
        print("  python scripts/analyze_curriculum_dataset.py --dataset Chip_v1_curriculum_stage1_n100")
        print("  python scripts/analyze_curriculum_dataset.py --all_stages")
        sys.exit(1)


if __name__ == "__main__":
    main()
