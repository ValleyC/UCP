from .BaseEnergy import BaseEnergyClass
from functools import partial
import jax
import jax.numpy as jnp
import jraph


class ChipPlacementEnergyClass(BaseEnergyClass):
    """
    Energy function for continuous chip placement optimization.

    The energy combines wirelength minimization with placement legality constraints:
        Energy = HPWL + overlap_weight * overlap_penalty + boundary_weight * boundary_penalty

    Components:
    - HPWL (Half-Perimeter Wirelength): Sum of bounding box perimeters for all nets
    - Overlap penalty: Smooth pairwise penalty for component overlaps
    - Boundary penalty: Quadratic penalty for components exceeding canvas bounds

    Penalty formulations:
    - Overlap: h = (ReLU(-l))² / 4, where l is smooth max of pairwise distances
    - Boundary: h = (ReLU(|x - center| + size/2 - limit))² / 2

    These smooth, differentiable formulations enable gradient-based optimization.
    """

    def __init__(self, config):
        if "continuous_dim" in config:
            config["n_bernoulli_features"] = config["continuous_dim"]
        elif "n_bernoulli_features" not in config:
            config["n_bernoulli_features"] = 2

        super().__init__(config)

        self.continuous_dim = config.get("continuous_dim", 2)
        self.overlap_weight = config.get("overlap_weight", 10.0)
        self.boundary_weight = config.get("boundary_weight", 10.0)
        self.overlap_threshold = config.get("overlap_threshold", 0.1)
        self.boundary_threshold = config.get("boundary_threshold", 0.1)
        self.canvas_width = config.get("canvas_width", 2.0)
        self.canvas_height = config.get("canvas_height", 2.0)
        self.canvas_x_min = config.get("canvas_x_min", -1.0)
        self.canvas_y_min = config.get("canvas_y_min", -1.0)

        print(f"ChipPlacement Energy Function")
        print(f"  Dimensions: {self.continuous_dim}")
        print(f"  Overlap weight: {self.overlap_weight}")
        print(f"  Boundary weight: {self.boundary_weight}")
        print(f"  Overlap threshold: {self.overlap_threshold}")
        print(f"  Boundary threshold: {self.boundary_threshold}")
        print(f"  Canvas: [{self.canvas_x_min}, {self.canvas_x_min + self.canvas_width}] x [{self.canvas_y_min}, {self.canvas_y_min + self.canvas_height}]")

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy(self, H_graph, positions, node_gr_idx, component_sizes=None):
        """
        Calculate total energy for chip placement.

        Args:
            H_graph: jraph graph structure containing:
                - nodes: component features (if any)
                - edges: netlist connectivity
                - senders: source component indices for edges
                - receivers: sink component indices for edges
                - n_node: number of components per graph
                - n_edge: number of nets per graph
            positions: component positions (shape: [num_components, continuous_dim])
                       For 2D: positions[:, 0] = x, positions[:, 1] = y
            node_gr_idx: mapping from components to graphs
            component_sizes: component sizes (shape: [num_components, 2]) for (x_size, y_size)
                           If None, extracted from graph nodes or assumed small default

        Returns:
            Energy_per_graph: total energy per graph (shape: [n_graphs, 1])
            positions: positions (for interface compatibility)
            constraint_violations_per_graph: sum of overlap + boundary violations (shape: [n_graphs, 1])
        """
        n_graph = H_graph.n_node.shape[0]
        nodes = H_graph.nodes
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]

        # Extract or default component sizes
        if component_sizes is None:
            # Try to extract from graph nodes (assuming nodes contain size information)
            # If nodes shape is [num_components, features] and features >= 2, use first 2 as sizes
            if nodes.shape[-1] >= 2:
                component_sizes = nodes[:, :2]  # First 2 features are x_size, y_size
            else:
                # Default: small components (0.1 x 0.1)
                component_sizes = jnp.full((total_num_nodes, 2), 0.1)

        # Ensure positions has correct shape [num_components, continuous_dim]
        if len(positions.shape) == 3:  # [num_components, 1, continuous_dim]
            positions = positions[:, 0, :]  # Remove middle dimension
        elif len(positions.shape) == 1:  # [num_components * continuous_dim]
            positions = jnp.reshape(positions, (total_num_nodes, self.continuous_dim))

        # 1. Compute HPWL (Half-Perimeter Wirelength)
        hpwl_per_graph = self._compute_hpwl(H_graph, positions, node_gr_idx, n_graph)

        # 2. Compute overlap penalty
        overlap_per_graph = self._compute_overlap_penalty(
            positions, component_sizes, node_gr_idx, n_graph
        )

        # 3. Compute boundary penalty
        boundary_per_graph = self._compute_boundary_penalty(
            positions, component_sizes, node_gr_idx, n_graph
        )

        Energy_per_graph = (
            hpwl_per_graph +
            self.overlap_weight * overlap_per_graph +
            self.boundary_weight * boundary_per_graph
        )

        # Constraint violations (for monitoring)
        constraint_violations_per_graph = overlap_per_graph + boundary_per_graph

        # Ensure output shape matches expected format [n_graphs, 1]
        if len(Energy_per_graph.shape) == 1:
            Energy_per_graph = jnp.expand_dims(Energy_per_graph, axis=-1)
        if len(constraint_violations_per_graph.shape) == 1:
            constraint_violations_per_graph = jnp.expand_dims(constraint_violations_per_graph, axis=-1)

        return Energy_per_graph, positions, constraint_violations_per_graph

    @partial(jax.jit, static_argnums=(0, 4))
    def _compute_hpwl(self, H_graph, positions, node_gr_idx, n_graph):
        """
        Compute Half-Perimeter Wirelength.

        HPWL = sum over all nets of (bbox_width + bbox_height)

        For each net (edge in graph):
        - Find bounding box of all connected components
        - HPWL contribution = width + height of bbox

        Args:
            H_graph: graph structure
            positions: component positions [num_components, 2]
            node_gr_idx: component to graph mapping
            n_graph: number of graphs

        Returns:
            hpwl_per_graph: HPWL per graph [n_graphs,]
        """
        senders = H_graph.senders
        receivers = H_graph.receivers

        # Get positions of sender and receiver components
        sender_pos = positions[senders]  # [num_edges, 2]
        receiver_pos = positions[receivers]  # [num_edges, 2]

        # For each edge, compute bounding box
        # Note: In chip placement, edges represent nets connecting two terminals
        # For 2-pin nets, bbox is simply the rectangle between the two points
        x_coords = jnp.stack([sender_pos[:, 0], receiver_pos[:, 0]], axis=1)  # [num_edges, 2]
        y_coords = jnp.stack([sender_pos[:, 1], receiver_pos[:, 1]], axis=1)  # [num_edges, 2]

        bbox_width = jnp.max(x_coords, axis=1) - jnp.min(x_coords, axis=1)  # [num_edges,]
        bbox_height = jnp.max(y_coords, axis=1) - jnp.min(y_coords, axis=1)  # [num_edges,]

        # HPWL per edge
        hpwl_per_edge = bbox_width + bbox_height  # [num_edges,]

        # Aggregate to graph level
        # Map edges to graphs via sender components
        edge_gr_idx = node_gr_idx[senders]
        hpwl_per_graph = jax.ops.segment_sum(hpwl_per_edge, edge_gr_idx, n_graph)

        return hpwl_per_graph

    @partial(jax.jit, static_argnums=(0, 4))
    def _compute_overlap_penalty(self, positions, component_sizes, node_gr_idx, n_graph, softmax_factor=10.0):
        """
        Compute smooth pairwise overlap penalty between components.

        For each component pair (i, j):
            delta = |pos_i - pos_j| - (size_i + size_j)/2
            l = sum(softmax(delta) * delta)
            h = (ReLU(-l))² / 4

        Penalties are weighted by component mass ratios for balanced gradient distribution.

        Args:
            positions: Component positions [num_components, 2]
            component_sizes: Component sizes [num_components, 2]
            node_gr_idx: Component to graph mapping
            n_graph: Number of graphs
            softmax_factor: Temperature for smooth max approximation

        Returns:
            overlap_per_graph: Total overlap penalty per graph [n_graphs,]
        """
        num_components = positions.shape[0]

        # Expand dimensions for pairwise computation
        positions_i = positions[:, jnp.newaxis, :]  # [N, 1, 2]
        positions_j = positions[jnp.newaxis, :, :]  # [1, N, 2]
        sizes_i = component_sizes[:, jnp.newaxis, :]  # [N, 1, 2]
        sizes_j = component_sizes[jnp.newaxis, :, :]  # [1, N, 2]

        delta = jnp.abs(positions_i - positions_j) - (sizes_i + sizes_j) / 2.0

        softmax_weights = jax.nn.softmax(delta * softmax_factor, axis=-1)
        l = jnp.sum(softmax_weights * delta, axis=-1)

        h = (jnp.maximum(0.0, -l) ** 2) / 4.0

        # Weight by geometric mean of component dimensions
        mass_i = jnp.exp(jnp.mean(jnp.log(sizes_i + 1e-8), axis=-1))  # [N, 1]
        mass_j = jnp.exp(jnp.mean(jnp.log(sizes_j + 1e-8), axis=-1))  # [1, N]
        mass_weight = mass_j / (mass_i + mass_j + 1e-8)  # [N, N]
        h_weighted = h * mass_weight  # [N, N]

        # Only count each pair once (upper triangle, excluding diagonal)
        i_indices = jnp.arange(num_components)[:, jnp.newaxis]
        j_indices = jnp.arange(num_components)[jnp.newaxis, :]
        upper_triangle_mask = (i_indices < j_indices).astype(jnp.float32)

        # Also mask to only consider pairs within same graph
        same_graph_mask = (node_gr_idx[:, jnp.newaxis] == node_gr_idx[jnp.newaxis, :]).astype(jnp.float32)

        # Combined mask
        valid_pairs_mask = upper_triangle_mask * same_graph_mask

        # Apply mask
        h_masked = h_weighted * valid_pairs_mask

        # Sum over all pairs per component, then aggregate to graph
        overlap_per_component = jnp.sum(h_masked, axis=1)  # [N,]
        overlap_per_graph = jax.ops.segment_sum(overlap_per_component, node_gr_idx, n_graph)

        return overlap_per_graph

    @partial(jax.jit, static_argnums=(0, 4))
    def _compute_boundary_penalty(self, positions, component_sizes, node_gr_idx, n_graph):
        """
        Compute quadratic boundary violation penalty.

        Penalizes components whose edges exceed canvas bounds:
            h_bound = (ReLU(|x - center| + size/2 - limit))² / 2

        Args:
            positions: Component positions [num_components, 2]
            component_sizes: Component sizes [num_components, 2]
            node_gr_idx: Component to graph mapping
            n_graph: Number of graphs

        Returns:
            boundary_penalty_per_graph: Total boundary penalty per graph [n_graphs,]
        """
        # Canvas boundaries and center
        canvas_x_max = self.canvas_x_min + self.canvas_width
        canvas_y_max = self.canvas_y_min + self.canvas_height
        canvas_center_x = (self.canvas_x_min + canvas_x_max) / 2.0
        canvas_center_y = (self.canvas_y_min + canvas_y_max) / 2.0
        canvas_half_width = self.canvas_width / 2.0
        canvas_half_height = self.canvas_height / 2.0

        dist_from_center_x = jnp.abs(positions[:, 0] - canvas_center_x)
        dist_from_center_y = jnp.abs(positions[:, 1] - canvas_center_y)

        half_sizes = component_sizes / 2.0

        violation_x = jnp.maximum(0.0, dist_from_center_x + half_sizes[:, 0] - canvas_half_width)
        violation_y = jnp.maximum(0.0, dist_from_center_y + half_sizes[:, 1] - canvas_half_height)

        boundary_penalty_per_component = ((violation_x ** 2) + (violation_y ** 2)) / 2.0
        boundary_penalty_per_graph = jax.ops.segment_sum(
            boundary_penalty_per_component, node_gr_idx, n_graph
        )

        return boundary_penalty_per_graph

    def calculate_relaxed_Energy(self, H_graph, positions, node_gr_idx, component_sizes=None):
        """
        Calculate relaxed energy (same as regular energy for continuous case).

        In discrete problems, this might use soft assignments. For continuous chip placement,
        we don't need a separate relaxed version.
        """
        return self.calculate_Energy(H_graph, positions, node_gr_idx, component_sizes)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_loss(self, H_graph, mean, log_var, node_gr_idx, component_sizes=None):
        """
        Calculate energy for training optimization.

        Energy = HPWL + overlap_weight × (overlap/threshold)² + boundary_weight × (boundary/threshold)²

        Args:
            H_graph: Graph structure
            mean: Predicted component positions
            log_var: Predicted log variance
            node_gr_idx: Component to graph mapping
            component_sizes: Component dimensions

        Returns:
            Energy, positions, constraint_violations
        """
        n_graph = H_graph.n_node.shape[0]
        nodes = H_graph.nodes
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]

        if component_sizes is None:
            if nodes.shape[-1] >= 2:
                component_sizes = nodes[:, :2]
            else:
                component_sizes = jnp.full((total_num_nodes, 2), 0.1)

        positions = mean
        if len(positions.shape) == 3:
            positions = positions[:, 0, :]
        elif len(positions.shape) == 1:
            positions = jnp.reshape(positions, (total_num_nodes, self.continuous_dim))

        hpwl_per_graph = self._compute_hpwl(H_graph, positions, node_gr_idx, n_graph)

        overlap_per_graph = self._compute_overlap_penalty(
            positions, component_sizes, node_gr_idx, n_graph
        )
        boundary_per_graph = self._compute_boundary_penalty(
            positions, component_sizes, node_gr_idx, n_graph
        )

        Energy_per_graph = (
            hpwl_per_graph +
            self.overlap_weight * ((overlap_per_graph / self.overlap_threshold) ** 2) +
            self.boundary_weight * ((boundary_per_graph / self.boundary_threshold) ** 2)
        )

        constraint_violations_per_graph = overlap_per_graph + boundary_per_graph

        if len(Energy_per_graph.shape) == 1:
            Energy_per_graph = jnp.expand_dims(Energy_per_graph, axis=-1)
        if len(constraint_violations_per_graph.shape) == 1:
            constraint_violations_per_graph = jnp.expand_dims(constraint_violations_per_graph, axis=-1)

        return Energy_per_graph, positions, constraint_violations_per_graph

    def get_HPWL_value(self, H_graph, positions, node_gr_idx):
        """
        Get HPWL value only (no penalties).

        Useful for evaluation and reporting.

        Args:
            H_graph: graph structure
            positions: component positions
            node_gr_idx: component to graph mapping

        Returns:
            HPWL per graph
        """
        n_graph = H_graph.n_node.shape[0]

        # Ensure positions has correct shape
        if len(positions.shape) == 3:
            positions = positions[:, 0, :]
        elif len(positions.shape) == 1:
            total_num_nodes = jax.tree_util.tree_leaves(H_graph.nodes)[0].shape[0]
            positions = jnp.reshape(positions, (total_num_nodes, self.continuous_dim))

        hpwl_per_graph = self._compute_hpwl(H_graph, positions, node_gr_idx, n_graph)

        return jnp.expand_dims(hpwl_per_graph, axis=-1)
