#!/usr/bin/env python
"""
Simple script to generate chip placement datasets

Usage:
    python generate_chip_dataset.py --dataset Chip_small --seed 123
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from DatasetCreator.loadGraphDatasets.ChipDatasetGenerator_Unsupervised import ChipDatasetGenerator

parser = argparse.ArgumentParser(description='Generate chip placement dataset')
parser.add_argument('--dataset', default='Chip_small',
                   choices=['Chip_v1', 'Chip_small', 'Chip_medium', 'Chip_large', 'Chip_huge',
                           'Chip_20_components', 'Chip_50_components', 'Chip_100_components',
                           'Chip_dummy'],
                   help='Dataset name (Chip_v1=ChipDiffusion v1 (~230 components), Chip_small=~10, Chip_medium=~20, Chip_large=~40, Chip_huge=~100 components)')
parser.add_argument('--seed', default=123, type=int, help='Random seed')
parser.add_argument('--modes', default=['train', 'val', 'test'], nargs='+',
                   help='Which splits to generate (default: train val test)')
parser.add_argument('--workers', default=None, type=int,
                   help='Number of parallel workers (default: cpu_count - 1)')
args = parser.parse_args()

def generate_dataset(dataset_name, seed, modes, n_workers=None):
    """Generate chip placement dataset for specified modes"""

    base_config = {
        'dataset': dataset_name,
        'seed': seed,
        'save': True,
        'dataset_name': dataset_name,
        'problem': 'ChipPlacement',
        'diff_ps': False,
        'gurobi_solve': False,
        'licence_base_path': '',
        'time_limit': 0,
        'thread_fraction': 1.0,
        'parent': False
    }

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Generating {mode} dataset for {dataset_name} with seed {seed}")
        print(f"{'='*60}\n")

        config = base_config.copy()
        config['mode'] = mode

        generator = ChipDatasetGenerator(config)
        generator.generate_dataset(n_workers=n_workers)

        print(f"\n✓ {mode} dataset generation complete!\n")

    print(f"\n{'='*60}")
    print(f"All datasets generated successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    generate_dataset(args.dataset, args.seed, args.modes, n_workers=args.workers)
