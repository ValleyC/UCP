"""
Chip Placement Dataset Generator for UNSUPERVISED Learning (SDDS)
Uses ORIGINAL ChipDiffusion proximity-based netlist generation

Method:
1. Place components in initial legal pattern
2. Generate netlist based on PROXIMITY (nearby components connect)
3. Randomize placement (shuffle positions)

Key properties:
- Netlist has meaningful spatial structure (proximity-based)
- Initial placement is randomized (high HPWL)
- Model learns to reduce HPWL by respecting netlist structure
"""
import torch
import torch.distributions as dist
import shapely
import numpy as np
import networkx as nx
from .BaseDatasetGenerator import BaseDatasetGenerator
from tqdm import tqdm
from torch_geometric.data import Data


class ChipDatasetGenerator(BaseDatasetGenerator):
    """
    Generates chip placement datasets for UNSUPERVISED learning
    Follows TSP pattern: fixed problem (netlist) + simple solution (random legal placement)
    """

    def __init__(self, config):
        super().__init__(config)

        # Default parameters
        self.max_instance = config.get("max_instance", 400)
        self.max_attempts_per_instance = config.get("max_attempts_per_instance", 1000)

        # Dataset sizes
        self.graph_config = {
            "n_train": config.get("n_train", 4000),
            "n_val": config.get("n_val", 500),
            "n_test": config.get("n_test", 1000),
        }

        if "huge" in self.dataset_name or "giant" in self.dataset_name:
            self.graph_config["n_test"] = 100

        if "dummy" in self.dataset_name:
            self.graph_config["n_train"] = 10
            self.graph_config["n_val"] = 5
            self.graph_config["n_test"] = 5

        print(f'\nGenerating Chip Placement {self.mode} dataset "{self.dataset_name}" '
              f'with {self.graph_config[f"n_{self.mode}"]} instances!')
        print(f'DATA GENERATION MODE: Unsupervised (netlist first, then random placement)\n')

    def generate_dataset(self):
        """Generate chip placement dataset instances"""
        solutions = {
            "positions": [],
            "H_graphs": [],
            "sizes": [],
            "edge_attrs": [],
            "graph_sizes": [],
            "densities": [],
            "Energies": [],  # HPWL
            "compl_H_graphs": [],
            "gs_bins": [],  # Placeholder for compatibility (continuous doesn't have ground truth bins)
        }

        for idx in tqdm(range(self.graph_config[f"n_{self.mode}"])):
            # Generate one chip placement instance
            positions, jraph_data, density, hpwl = self.sample_chip_instance_unsupervised()

            solutions["positions"].append(positions.numpy())
            solutions["H_graphs"].append(jraph_data)
            solutions["sizes"].append(jraph_data.nodes)  # Already numpy from jraph
            solutions["edge_attrs"].append(jraph_data.edges)  # Already numpy from jraph
            solutions["graph_sizes"].append(positions.shape[0])
            solutions["densities"].append(density)
            solutions["Energies"].append(hpwl)
            solutions["compl_H_graphs"].append(jraph_data)
            # For continuous problems, we store positions as "gs_bins" (though semantically different)
            # This maintains compatibility with the data loader
            solutions["gs_bins"].append(positions.numpy())

            # Save individual instance
            indexed_solution_dict = {}
            for key in solutions.keys():
                if len(solutions[key]) > 0:
                    indexed_solution_dict[key] = solutions[key][idx]
            self.save_instance_solution(indexed_solution_dict, idx)

        # Save all solutions
        self.save_solutions(solutions)

    def sample_chip_instance_unsupervised(self):
        """
        Sample one chip placement instance for UNSUPERVISED learning

        ChipDiffusion Method:
        1. Place components in initial pattern
        2. Generate netlist based on PROXIMITY (nearby components connect)
        3. Randomize placement (shuffle positions)

        Returns:
            positions: (V, 2) tensor of component center positions
            data: PyG Data object with graph structure
            density: float, placement density
            hpwl: float, Half-Perimeter Wirelength
        """

        # 1. Sample target density
        stop_density = self._sample_uniform(0.75, 0.9)

        # Component generation: Control number via pool size and component sizes
        # For ~8-12 components with large sizes: max_instance=25, long_sizes=[0.4, 0.75]
        # For ~15-25 components with smaller sizes: max_instance=60, long_sizes=[0.2, 0.5]
        # For ~30-50 components with tiny sizes: max_instance=120, long_sizes=[0.1, 0.35]

        # Parse dataset name to determine scale
        if "small" in self.dataset_name.lower():
            # Small: ~8-12 components, larger sizes
            max_instance = 25
            long_size_range = (0.4, 0.75)
        elif "medium" in self.dataset_name.lower() or "20" in self.dataset_name:
            # Medium: ~15-25 components, medium sizes
            max_instance = 60
            long_size_range = (0.2, 0.5)
        elif "large" in self.dataset_name.lower() or "50" in self.dataset_name:
            # Large: ~30-50 components, small sizes
            max_instance = 120
            long_size_range = (0.1, 0.35)
        elif "huge" in self.dataset_name.lower() or "100" in self.dataset_name:
            # Huge: ~60-100 components, tiny sizes
            max_instance = 250
            long_size_range = (0.08, 0.25)
        else:
            # Default: small
            max_instance = 25
            long_size_range = (0.4, 0.75)

        aspect_ratios = self._sample_uniform(0.4, 1.0, (max_instance,))  # More square-ish
        # Use UNIFORM distribution for consistently sized components
        long_sizes = self._sample_uniform(long_size_range[0], long_size_range[1], (max_instance,))
        short_sizes = aspect_ratios * long_sizes

        # Random orientation
        long_x = (torch.rand(max_instance) > 0.5).float()
        x_sizes = long_x * long_sizes + (1 - long_x) * short_sizes
        y_sizes = (1 - long_x) * long_sizes + long_x * short_sizes

        # 3. FIRST: Place components until density target is reached
        initial_positions, placed_sizes, actual_density = self._place_components_legal(
            x_sizes, y_sizes, stop_density
        )

        # 4. SECOND: Generate netlist based on PROXIMITY in initial placement
        edge_index, edge_attr = self._generate_netlist_proximity_based(
            initial_positions, placed_sizes
        )

        # 5. THIRD: Randomize placement with completely NEW random positions
        # Generate uniform random positions within canvas bounds [-1, 1]
        # No legality constraint needed - overlaps are okay for random starting point
        # This destroys all spatial relationships from the legal placement
        randomized_positions = torch.rand(len(placed_sizes), 2) * 2 - 1  # Uniform in [-1, 1]

        # 6. Compute HPWL (will be high because placement is randomized!)
        hpwl = self._compute_hpwl(randomized_positions, placed_sizes, edge_index, edge_attr)

        # 7. Create PyG Data object
        data = Data(
            x=placed_sizes,           # (V, 2) component sizes
            edge_index=edge_index,    # (2, E)
            edge_attr=edge_attr,      # (E, 4) terminal offsets
            is_ports=torch.zeros(len(placed_sizes), dtype=torch.bool)
        )

        # 8. Convert to Jraph format for SDDS compatibility
        jraph_data = self._pyg_to_jraph(data, randomized_positions)

        return randomized_positions, jraph_data, actual_density, hpwl

    def _generate_netlist_proximity_based(self, positions, sizes):
        """
        Generate netlist based on SPATIAL PROXIMITY (ChipDiffusion method)

        Components that are close together in the initial placement get connected.
        This creates meaningful spatial structure in the netlist.

        Args:
            positions: (V, 2) tensor of component center positions
            sizes: (V, 2) tensor of component sizes

        Returns:
            edge_index: (2, E) tensor
            edge_attr: (E, 4) tensor of terminal offsets
        """
        num_components = positions.shape[0]

        if num_components < 2:
            # Fallback for single component
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.zeros((1, 4), dtype=torch.float32)
            return edge_index, edge_attr

        # Compute pairwise distances between component centers
        dist_matrix = torch.cdist(positions, positions, p=2)  # (V, V) Euclidean distances

        # K-nearest neighbors approach (connect to k nearest neighbors)
        k = min(5, num_components - 1)  # Connect to 3-5 nearest neighbors

        edge_list = []
        for i in range(num_components):
            # Get k nearest neighbors (excluding self)
            distances = dist_matrix[i, :]
            distances[i] = float('inf')  # Exclude self
            _, nearest_indices = torch.topk(distances, k, largest=False)

            for j in nearest_indices:
                edge_list.append([i, j.item()])

        # Generate terminals for each component
        x_sizes = sizes[:, 0]
        y_sizes = sizes[:, 1]
        areas = x_sizes * y_sizes

        num_terminals = self._sample_conditional_binomial(
            areas,
            binom_p=0.5,
            binom_min_n=2,
            t=64,
            p=0.65
        )
        num_terminals = torch.clamp(num_terminals, min=2, max=128).int()
        max_num_terminals = torch.max(num_terminals)

        # Terminal offsets relative to component center
        terminal_offsets = self._get_terminal_offsets(x_sizes, y_sizes, max_num_terminals)

        # Create edge attributes with terminal assignments
        edge_attrs = []
        for (u, v) in edge_list:
            # Select random terminals from each component
            term_u = np.random.randint(0, num_terminals[u].item())
            term_v = np.random.randint(0, num_terminals[v].item())

            # Get terminal offsets
            offset_u = terminal_offsets[u, term_u, :]  # (2,)
            offset_v = terminal_offsets[v, term_v, :]  # (2,)

            edge_attrs.append(torch.cat([offset_u, offset_v]))

        if len(edge_list) == 0:
            # Fallback: create at least one edge
            edge_list = [[0, min(1, num_components-1)]]
            offset_0 = terminal_offsets[0, 0, :]
            offset_1 = terminal_offsets[min(1, num_components-1), 0, :]
            edge_attrs = [torch.cat([offset_0, offset_1])]

        edge_index = torch.tensor(edge_list, dtype=torch.long).T  # (2, E)
        edge_attr = torch.stack(edge_attrs)  # (E, 4)

        return edge_index, edge_attr

    def _randomize_placement(self, sizes):
        """
        Randomize component placement (shuffle positions)

        This destroys the spatial relationships in the original placement,
        creating a high HPWL that the model must learn to reduce.

        Args:
            sizes: (V, 2) tensor of component sizes

        Returns:
            positions: (V, 2) tensor of randomized center positions
        """
        num_components = sizes.shape[0]
        x_sizes = sizes[:, 0]
        y_sizes = sizes[:, 1]

        placement = ChipPlacement()

        # Shuffle the order of placement
        indices = torch.randperm(num_components)

        for idx in indices:
            x_size_val = float(x_sizes[idx])
            y_size_val = float(y_sizes[idx])

            # Calculate valid position range
            low = torch.tensor([(x_size_val/2) - 1.0, (y_size_val/2) - 1.0])
            high = torch.tensor([1.0 - (x_size_val/2), 1.0 - (y_size_val/2)])

            # Check if component fits
            if (low >= high).any():
                continue

            placed = False
            for attempt in range(self.max_attempts_per_instance):
                # Random position (completely independent of netlist)
                candidate_pos = torch.rand(2) * (high - low) + low

                if placement.check_legality(candidate_pos[0].item(),
                                           candidate_pos[1].item(),
                                           x_size_val, y_size_val):
                    placement.commit_instance(candidate_pos[0].item(),
                                            candidate_pos[1].item(),
                                            x_size_val, y_size_val)
                    placed = True
                    break

            if not placed:
                # Fallback: place at a default position if couldn't find legal spot
                # This shouldn't happen often, but prevents crashes
                default_pos = torch.tensor([0.0, 0.0])
                placement.commit_instance(default_pos[0].item(),
                                        default_pos[1].item(),
                                        x_size_val, y_size_val)

        positions = placement.get_positions()
        return positions

    def _place_components_legal(self, x_sizes, y_sizes, stop_density):
        """
        Place components randomly (legal but NOT HPWL-optimized)
        This is the "simple solution" analogous to TSP's sequential tour.

        Args:
            x_sizes: (V,) tensor
            y_sizes: (V,) tensor
            stop_density: float, target density

        Returns:
            positions: (V, 2) tensor
            sizes: (V, 2) tensor (actually placed)
            density: float (actual density achieved)
        """
        placement = ChipPlacement()
        density = 0.0

        # Sort by area (largest first for better packing success rate)
        areas = x_sizes * y_sizes
        _, indices = torch.sort(areas, descending=True)
        x_sizes_sorted = x_sizes[indices]
        y_sizes_sorted = y_sizes[indices]

        placed_components = []

        for idx, (x_size, y_size) in enumerate(zip(x_sizes_sorted, y_sizes_sorted)):
            x_size_val = float(x_size)
            y_size_val = float(y_size)

            # Calculate valid position range
            low = torch.tensor([(x_size_val/2) - 1.0, (y_size_val/2) - 1.0])
            high = torch.tensor([1.0 - (x_size_val/2), 1.0 - (y_size_val/2)])

            # Check if component fits
            if (low >= high).any():
                continue  # Component too large, skip

            placed = False
            for attempt in range(self.max_attempts_per_instance):
                # Random position (don't consider netlist connectivity!)
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

            if density >= stop_density:
                break

        # Extract placement data
        positions = placement.get_positions()
        sizes = placement.get_sizes()

        return positions, sizes, density

    # ========== Helper Methods (from original) ==========

    def _sample_uniform(self, low, high, size=None):
        """Sample from uniform distribution"""
        if size is None:
            return torch.rand(1).item() * (high - low) + low
        return torch.rand(*size) * (high - low) + low

    def _sample_clipped_exp(self, scale, low, high, size):
        """Sample from clipped exponential distribution"""
        samples = torch.empty(size).exponential_(lambd=1.0/scale)
        return torch.clamp(samples, min=low, max=high)

    def _sample_conditional_binomial(self, areas, binom_p, binom_min_n, t, p):
        """Sample number of terminals based on component area"""
        n = torch.clamp(torch.ceil(t * torch.pow(areas, p)), min=binom_min_n)
        distribution = dist.Binomial(n, torch.full(areas.shape, binom_p))
        return distribution.sample().int()

    def _get_terminal_offsets(self, x_sizes, y_sizes, max_num_terminals):
        """
        Generate terminal offsets on component boundaries

        Returns:
            terminal_offsets: (V, T, 2) tensor
        """
        half_perim = x_sizes + y_sizes

        # Sample locations along perimeter
        terminal_locs = torch.rand(max_num_terminals, x_sizes.shape[0]) * half_perim
        terminal_flip = (torch.rand(max_num_terminals, x_sizes.shape[0]) > 0.5).float()
        terminal_flip = 2 * terminal_flip - 1

        x_sizes_exp = x_sizes.unsqueeze(0)
        y_sizes_exp = y_sizes.unsqueeze(0)

        # Map to x/y offsets
        offset_x = torch.clamp(terminal_locs, torch.zeros_like(x_sizes_exp),
                              x_sizes_exp) - (x_sizes_exp / 2)
        offset_y = torch.clamp(terminal_locs - x_sizes_exp, torch.zeros_like(y_sizes_exp),
                              y_sizes_exp) - (y_sizes_exp / 2)

        offset_x = terminal_flip * offset_x
        offset_y = terminal_flip * offset_y

        terminal_offset = torch.stack((offset_x, offset_y), dim=-1)  # (T, V, 2)
        terminal_offset = terminal_offset.permute(1, 0, 2)  # (V, T, 2)

        return terminal_offset

    def _compute_hpwl(self, positions, sizes, edge_index, edge_attr):
        """
        Compute Half-Perimeter Wirelength

        HPWL = Σ_net (bbox_width + bbox_height)

        This is the EXACT method from ChipDiffusion for compatibility.
        """
        if edge_index.shape[1] == 0:
            return 0.0

        # Get unique nets (undirected)
        num_edges = edge_index.shape[1] // 2
        edge_index_unique = edge_index[:, :num_edges]
        edge_attr_unique = edge_attr[:num_edges, :]

        # Compute terminal positions
        src_idx = edge_index_unique[0, :]
        sink_idx = edge_index_unique[1, :]

        src_term_pos = positions[src_idx] + edge_attr_unique[:, :2]
        sink_term_pos = positions[sink_idx] + edge_attr_unique[:, 2:4]

        # Compute bounding box per net
        # Simplified: treat each 2-pin connection as a net
        x_coords = torch.stack([src_term_pos[:, 0], sink_term_pos[:, 0]], dim=1)
        y_coords = torch.stack([src_term_pos[:, 1], sink_term_pos[:, 1]], dim=1)

        bbox_width = x_coords.max(dim=1)[0] - x_coords.min(dim=1)[0]
        bbox_height = y_coords.max(dim=1)[0] - y_coords.min(dim=1)[0]

        hpwl = (bbox_width + bbox_height).sum().item()
        return hpwl

    def _pyg_to_jraph(self, pyg_data, positions):
        """
        Convert PyTorch Geometric Data to Jraph GraphsTuple for SDDS compatibility.

        Args:
            pyg_data: PyG Data object
            positions: (V, 2) tensor of positions (not used in graph structure, but stored in nodes)

        Returns:
            jraph_graph: Jraph GraphsTuple
        """
        import jraph

        # Extract data from PyG
        num_nodes = pyg_data.x.shape[0]
        num_edges = pyg_data.edge_index.shape[1]

        # Convert to numpy for jraph
        # Nodes: Store component sizes (x_size, y_size)
        nodes = pyg_data.x.numpy().astype(np.float32)

        # Edges: Store terminal offsets (src_x_offset, src_y_offset, sink_x_offset, sink_y_offset)
        edges = pyg_data.edge_attr.numpy().astype(np.float32)

        # Senders and receivers
        senders = pyg_data.edge_index[0, :].numpy().astype(np.int32)
        receivers = pyg_data.edge_index[1, :].numpy().astype(np.int32)

        # Create jraph GraphsTuple
        jraph_graph = jraph.GraphsTuple(
            nodes=nodes,                      # (V, 2) component sizes
            edges=edges,                      # (E, 4) terminal offsets
            senders=senders,                  # (E,) source component indices
            receivers=receivers,              # (E,) target component indices
            n_node=np.array([num_nodes]),     # Number of nodes per graph
            n_edge=np.array([num_edges]),     # Number of edges per graph
            globals=None                      # No global features
        )

        return jraph_graph


class ChipPlacement:
    """
    Helper class for managing chip placement with collision detection
    Uses shapely for efficient geometric operations
    """

    def __init__(self):
        self.instances = []
        self.x_coords = []
        self.y_coords = []
        self.x_sizes = []
        self.y_sizes = []
        self.is_port = []

        self.chip_bounds = shapely.box(-1, -1, 1, 1)
        self.eps = 1e-8

    def check_legality(self, x_pos, y_pos, x_size, y_size):
        """
        Check if placing component at (x_pos, y_pos) is legal
        Positions are center coordinates
        """
        candidate = shapely.box(
            x_pos - x_size/2, y_pos - y_size/2,
            x_pos + x_size/2, y_pos + y_size/2
        )

        # Check chip boundary
        if not self.chip_bounds.contains(candidate):
            return False

        # Check overlap with existing instances
        for inst in self.instances:
            if inst.intersects(candidate):
                return False

        return True

    def commit_instance(self, x_pos, y_pos, x_size, y_size, is_port=False):
        """Add instance to placement (assumes center coordinates)"""
        if not is_port:
            self.instances.append(shapely.box(
                x_pos - x_size/2, y_pos - y_size/2,
                x_pos + x_size/2, y_pos + y_size/2
            ))

        self.x_coords.append(x_pos)
        self.y_coords.append(y_pos)
        self.x_sizes.append(x_size)
        self.y_sizes.append(y_size)
        self.is_port.append(is_port)

    def get_positions(self):
        """Return (V, 2) tensor of center positions"""
        return torch.stack([
            torch.tensor(self.x_coords),
            torch.tensor(self.y_coords)
        ], dim=-1)

    def get_sizes(self):
        """Return (V, 2) tensor of sizes"""
        return torch.stack([
            torch.tensor(self.x_sizes),
            torch.tensor(self.y_sizes)
        ], dim=-1)

    def get_mask(self):
        """Return (V,) boolean tensor for ports"""
        return torch.tensor(self.is_port)

    def get_density(self):
        """Compute placement density"""
        total_area = sum([inst.area for inst in self.instances])
        return total_area / self.chip_bounds.area
