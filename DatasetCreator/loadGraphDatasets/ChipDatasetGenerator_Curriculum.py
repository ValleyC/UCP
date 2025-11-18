"""
Chip Placement Dataset Generator for CURRICULUM LEARNING

Extended from ChipDatasetGenerator_Unsupervised to support:
- Target component count ranges (e.g., 90-110 components)
- Fixed-size datasets for staged training

Usage:
    Generate stage 1 (100 components):
        python generate_chip_curriculum.py --stage 1 --target_components 100 --component_variation 10

    Generate stage 2 (200 components):
        python generate_chip_curriculum.py --stage 2 --target_components 200 --component_variation 20
"""

from .ChipDatasetGenerator_Unsupervised import ChipDatasetGenerator as BaseGenerator
import torch
import numpy as np


class ChipDatasetGeneratorCurriculum(BaseGenerator):
    """
    Curriculum-aware chip placement dataset generator

    Extends base generator to control component count distribution
    """

    def __init__(self, config):
        super().__init__(config)

        # Curriculum parameters
        self.target_components = config.get("target_components", None)  # e.g., 100
        self.component_variation = config.get("component_variation", 10)  # ±10 variation

        if self.target_components is not None:
            self.component_min = max(10, self.target_components - self.component_variation)
            self.component_max = self.target_components + self.component_variation
            print(f'\nCURRICULUM MODE: Target {self.target_components} components '
                  f'(range: [{self.component_min}, {self.component_max}])')
        else:
            # Original v1 behavior
            self.component_min = None
            self.component_max = None
            print(f'\nSTANDARD MODE: Variable component count (original v1 behavior)')

    def sample_chip_instance_unsupervised(self):
        """
        Sample one chip placement instance with CONTROLLED component count

        Modified from base class to enforce target component range
        """

        # 1. Sample target density
        stop_density = self._sample_uniform(0.75, 0.9)

        # 2. Configure component generation based on target count
        if self.target_components is not None:
            # CURRICULUM MODE: Adjust parameters based on target
            component_count_target = np.random.randint(self.component_min, self.component_max + 1)
            max_instance, exp_scale, exp_min, exp_max = self._get_curriculum_params(
                component_count_target
            )
        else:
            # ORIGINAL v1 MODE
            max_instance = 400
            exp_scale = 0.08
            exp_min = 0.02
            exp_max = 1.0
            component_count_target = None

        aspect_ratio_range = (0.25, 1.0)

        # 3. Generate component pool
        aspect_ratios = self._sample_uniform(
            aspect_ratio_range[0], aspect_ratio_range[1], (max_instance,)
        )

        # Clipped Exponential size distribution (ChipDiffusion v1)
        long_sizes = self._sample_clipped_exp(exp_scale, exp_min, exp_max, (max_instance,))
        short_sizes = aspect_ratios * long_sizes

        # Random orientation
        long_x = (torch.rand(max_instance) > 0.5).float()
        x_sizes = long_x * long_sizes + (1 - long_x) * short_sizes
        y_sizes = (1 - long_x) * long_sizes + long_x * short_sizes

        # 4. Place components with target count enforcement
        if component_count_target is not None:
            initial_positions, placed_sizes, actual_density = self._place_components_with_target(
                x_sizes, y_sizes, stop_density, component_count_target
            )
        else:
            initial_positions, placed_sizes, actual_density = self._place_components_legal(
                x_sizes, y_sizes, stop_density
            )

        # 5. Generate netlist based on proximity
        edge_index, edge_attr = self._generate_netlist_proximity_based(
            initial_positions, placed_sizes
        )

        # 6. Randomize placement
        randomized_positions = torch.rand(len(placed_sizes), 2) * 2 - 1  # Uniform in [-1, 1]

        # 7. Compute HPWL
        hpwl = self._compute_hpwl(randomized_positions, placed_sizes, edge_index, edge_attr)

        # 8. Create jraph.GraphsTuple
        import jraph
        randomized_positions_np = randomized_positions.numpy() if isinstance(randomized_positions, torch.Tensor) else randomized_positions
        legal_positions_np = initial_positions.numpy() if isinstance(initial_positions, torch.Tensor) else initial_positions
        sizes_np = placed_sizes.numpy() if isinstance(placed_sizes, torch.Tensor) else placed_sizes
        edge_index_np = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index
        edge_attr_np = edge_attr.numpy() if isinstance(edge_attr, torch.Tensor) else edge_attr

        num_nodes = sizes_np.shape[0]
        num_edges = edge_index_np.shape[1]

        H_graph = jraph.GraphsTuple(
            nodes=sizes_np,
            edges=edge_attr_np,
            senders=edge_index_np[0, :],
            receivers=edge_index_np[1, :],
            n_node=np.array([num_nodes]),
            n_edge=np.array([num_edges]),
            globals=None
        )

        return randomized_positions_np, legal_positions_np, H_graph, actual_density, hpwl

    def _get_curriculum_params(self, target_count):
        """
        Dynamically adjust component pool and size distribution for target count

        Key insight: To reliably hit target count, we need:
        1. Smaller components (so more fit in the canvas)
        2. Larger pool (more candidates to choose from)
        3. Proper density targets

        Args:
            target_count: Desired number of components

        Returns:
            max_instance: Component pool size
            exp_scale: Exponential distribution scale
            exp_min: Min component size
            exp_max: Max component size
        """

        # Pool size: 3-4x target (ensures enough small components)
        max_instance = int(target_count * 3.5)

        # Critical fix: Scale component sizes based on target count
        #
        # Target area calculation:
        # - Canvas area = 4.0
        # - Target density ≈ 0.8 → total component area ≈ 3.2
        # - For N components: avg area per component ≈ 3.2/N
        #
        # Component area = long_size × short_size = long_size² × aspect_ratio
        # E[area] ≈ 1.25 × exp_scale² (accounting for aspect ratio distribution)
        #
        # Solving: 1.25 × β² = 3.2/N → β = sqrt(3.2/(1.25×N)) = sqrt(2.56/N)

        import math

        # Calculate ideal scale from target density
        ideal_scale = math.sqrt(2.56 / target_count)

        # Apply stage-specific adjustments
        if target_count < 100:
            # ~50-100 components
            exp_scale = max(0.12, ideal_scale * 1.1)  # Slightly larger for safety
            exp_min = 0.02
            exp_max = 0.8
        elif target_count < 150:
            # ~100-150 components
            exp_scale = max(0.10, ideal_scale)
            exp_min = 0.018
            exp_max = 0.6
        elif target_count < 200:
            # ~150-200 components
            exp_scale = max(0.08, ideal_scale * 0.95)
            exp_min = 0.015
            exp_max = 0.5
        elif target_count < 300:
            # ~200-300 components
            exp_scale = max(0.07, ideal_scale * 0.9)
            exp_min = 0.012
            exp_max = 0.4
        else:
            # 300+ components
            exp_scale = max(0.06, ideal_scale * 0.85)
            exp_min = 0.01
            exp_max = 0.35

        return max_instance, exp_scale, exp_min, exp_max

    def _place_components_with_target(self, x_sizes, y_sizes, stop_density, target_count):
        """
        Place components with DUAL stopping criteria:
        1. Density threshold (original)
        2. Component count threshold (NEW)

        Args:
            x_sizes: (V,) tensor
            y_sizes: (V,) tensor
            stop_density: float, target density (0.75-0.9)
            target_count: int, specific target for this instance (e.g., 95, 103)

        Returns:
            positions: (V, 2) tensor
            sizes: (V, 2) tensor
            density: float
        """
        from .ChipDatasetGenerator_Unsupervised import ChipPlacement

        placement = ChipPlacement()
        density = 0.0

        # Sort by area (LARGEST first to ensure proper density!)
        # Rationale:
        # - Large components contribute most to density
        # - If we place small ones first, we hit component count before density target
        # - Need to fill canvas with large pieces first, then pack small ones
        areas = x_sizes * y_sizes
        _, indices = torch.sort(areas, descending=True)  # LARGEST first!
        x_sizes_sorted = x_sizes[indices]
        y_sizes_sorted = y_sizes[indices]

        placed_components = []

        # Relaxed bounds: allow some flexibility to reach density target
        # Priority order:
        # 1. Reach minimum component count
        # 2. Reach density target
        # 3. Try to stay close to target_count (but density is more important!)
        min_components = max(10, target_count - 5)
        max_components = target_count + 10  # Allow up to +10 to reach density

        for idx, (x_size, y_size) in enumerate(zip(x_sizes_sorted, y_sizes_sorted)):
            x_size_val = float(x_size)
            y_size_val = float(y_size)

            # Calculate valid position range
            low = torch.tensor([(x_size_val/2) - 1.0, (y_size_val/2) - 1.0])
            high = torch.tensor([1.0 - (x_size_val/2), 1.0 - (y_size_val/2)])

            # Check if component fits
            if (low >= high).any():
                continue

            placed = False
            for attempt in range(self.max_attempts_per_instance):
                candidate_pos = torch.rand(2) * (high - low) + low

                if placement.check_legality(candidate_pos[0].item(),
                                           candidate_pos[1].item(),
                                           x_size_val, y_size_val):
                    placement.commit_instance(candidate_pos[0].item(),
                                            candidate_pos[1].item(),
                                            x_size_val, y_size_val)
                    placed_components.append(indices[idx].item())
                    placed = True
                    break

            if placed:
                density += (x_size_val * y_size_val) / 4.0

            num_placed = len(placed_components)

            # STOPPING CRITERIA (prioritize DENSITY over exact component count)
            # We want circuits with proper density (0.75-0.9) more than exact count
            #
            # Stop if:
            # 1. Minimum reached: Have min_components AND density >= target
            # 2. Good enough: In component range AND density >= 90% of target
            # 3. Safety: Exceeded max_components (but this shouldn't happen often)

            if num_placed >= min_components and density >= stop_density:
                # Success! We have enough components and proper density
                break

            if min_components <= num_placed <= max_components:
                # In component range - check if density is close enough
                if density >= stop_density * 0.9:  # Within 10% of density target
                    break

            if num_placed >= max_components:
                # Safety: don't exceed max (even if density low)
                break

        # Extract placement data
        positions = placement.get_positions()
        sizes = placement.get_sizes()

        return positions, sizes, density
