#!/usr/bin/env python
"""
Generate chip placement datasets for CURRICULUM LEARNING

Generates datasets with controlled component count ranges for staged training.

Usage Examples:
    # Stage 1: 100 components (±10)
    python generate_chip_curriculum.py --stage 1 --target_components 100 --component_variation 10 --seed 123

    # Stage 2: 150 components (±10)
    python generate_chip_curriculum.py --stage 2 --target_components 150 --component_variation 10 --seed 123

    # Stage 3: 200 components (±20)
    python generate_chip_curriculum.py --stage 3 --target_components 200 --component_variation 20 --seed 123

    # Generate all stages at once
    python generate_chip_curriculum.py --all_stages --seed 123

Dataset Statistics:
    After generation, check component distribution:
        python scripts/analyze_curriculum_dataset.py --dataset Chip_v1_curriculum_stage1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from DatasetCreator.loadGraphDatasets.ChipDatasetGenerator_Curriculum import ChipDatasetGeneratorCurriculum

# Predefined curriculum stages
CURRICULUM_STAGES = [
    {
        'stage': 1,
        'name': 'stage1_n100',
        'target_components': 100,
        'component_variation': 10,
        'description': 'Small circuits (90-110 components)'
    },
    {
        'stage': 2,
        'name': 'stage2_n150',
        'target_components': 150,
        'component_variation': 10,
        'description': 'Medium-small circuits (140-160 components)'
    },
    {
        'stage': 3,
        'name': 'stage3_n200',
        'target_components': 200,
        'component_variation': 20,
        'description': 'Medium circuits (180-220 components)'
    },
    {
        'stage': 4,
        'name': 'stage4_n250',
        'target_components': 250,
        'component_variation': 20,
        'description': 'Medium-large circuits (230-270 components)'
    },
    {
        'stage': 5,
        'name': 'stage5_n350',
        'target_components': 350,
        'component_variation': 30,
        'description': 'Large circuits (320-380 components)'
    }
]

parser = argparse.ArgumentParser(description='Generate curriculum chip placement dataset')
parser.add_argument('--stage', type=int, default=None,
                   help='Curriculum stage (1-5). Use --all_stages to generate all.')
parser.add_argument('--target_components', type=int, default=None,
                   help='Target number of components (overrides stage preset)')
parser.add_argument('--component_variation', type=int, default=10,
                   help='Variation range around target (±variation)')
parser.add_argument('--seed', default=123, type=int,
                   help='Random seed')
parser.add_argument('--modes', default=['train', 'val', 'test'], nargs='+',
                   help='Which splits to generate (default: train val test)')
parser.add_argument('--workers', default=None, type=int,
                   help='Number of parallel workers (default: cpu_count - 1)')
parser.add_argument('--all_stages', action='store_true',
                   help='Generate all 5 curriculum stages')
parser.add_argument('--n_train', type=int, default=1000,
                   help='Number of training instances')
parser.add_argument('--n_val', type=int, default=200,
                   help='Number of validation instances')
parser.add_argument('--n_test', type=int, default=300,
                   help='Number of test instances')
args = parser.parse_args()


def generate_curriculum_stage(stage_config, seed, modes, n_workers=None, dataset_sizes=None):
    """Generate a single curriculum stage"""

    dataset_name = f"Chip_v1_curriculum_{stage_config['name']}"

    print("\n" + "="*70)
    print(f"CURRICULUM STAGE {stage_config['stage']}: {stage_config['description']}")
    print(f"Dataset: {dataset_name}")
    print(f"Target components: {stage_config['target_components']} "
          f"(±{stage_config['component_variation']})")
    print("="*70 + "\n")

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
        'parent': False,
        'target_components': stage_config['target_components'],
        'component_variation': stage_config['component_variation']
    }

    # Add dataset sizes if provided
    if dataset_sizes:
        base_config.update(dataset_sizes)

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Generating {mode} split...")
        print(f"{'='*60}\n")

        config = base_config.copy()
        config['mode'] = mode

        generator = ChipDatasetGeneratorCurriculum(config)
        generator.generate_dataset(n_workers=n_workers)

        print(f"\n[OK] {mode} split complete!\n")

    print(f"\n{'='*70}")
    print(f"Stage {stage_config['stage']} generation complete!")
    print(f"Location: data/{dataset_name}/")
    print("="*70 + "\n")


def main():
    dataset_sizes = {
        'n_train': args.n_train,
        'n_val': args.n_val,
        'n_test': args.n_test
    }

    if args.all_stages:
        # Generate all curriculum stages
        print("\n" + "="*70)
        print("GENERATING ALL CURRICULUM STAGES")
        print("="*70)

        for stage_config in CURRICULUM_STAGES:
            generate_curriculum_stage(
                stage_config,
                args.seed,
                args.modes,
                n_workers=args.workers,
                dataset_sizes=dataset_sizes
            )

        print("\n" + "="*70)
        print("ALL CURRICULUM STAGES GENERATED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated datasets:")
        for stage_config in CURRICULUM_STAGES:
            print(f"  - Stage {stage_config['stage']}: "
                  f"Chip_v1_curriculum_{stage_config['name']} "
                  f"({stage_config['description']})")

    elif args.stage is not None:
        # Generate specific stage
        if not (1 <= args.stage <= len(CURRICULUM_STAGES)):
            print(f"Error: Stage must be between 1 and {len(CURRICULUM_STAGES)}")
            sys.exit(1)

        stage_config = CURRICULUM_STAGES[args.stage - 1]

        # Allow overriding preset values
        if args.target_components is not None:
            stage_config = stage_config.copy()
            stage_config['target_components'] = args.target_components
            stage_config['component_variation'] = args.component_variation
            stage_config['name'] = f"stage{args.stage}_n{args.target_components}"

        generate_curriculum_stage(
            stage_config,
            args.seed,
            args.modes,
            n_workers=args.workers,
            dataset_sizes=dataset_sizes
        )

    elif args.target_components is not None:
        # Custom stage with user-specified parameters
        stage_config = {
            'stage': 'custom',
            'name': f"custom_n{args.target_components}",
            'target_components': args.target_components,
            'component_variation': args.component_variation,
            'description': f'Custom ({args.target_components - args.component_variation}-'
                         f'{args.target_components + args.component_variation} components)'
        }

        generate_curriculum_stage(
            stage_config,
            args.seed,
            args.modes,
            n_workers=args.workers,
            dataset_sizes=dataset_sizes
        )

    else:
        print("Error: Must specify either --stage, --all_stages, or --target_components")
        print("\nExamples:")
        print("  python generate_chip_curriculum.py --stage 1")
        print("  python generate_chip_curriculum.py --all_stages")
        print("  python generate_chip_curriculum.py --target_components 100")
        sys.exit(1)


if __name__ == "__main__":
    main()
