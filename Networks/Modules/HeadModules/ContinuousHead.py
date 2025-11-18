import jax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
import flax


def get_graph_info(jraph_graph_list):
    """Extract graph structure information."""
    first_graph = jraph_graph_list["graphs"][0]
    nodes = first_graph.nodes
    n_node = first_graph.n_node
    n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
    graph_idx = jnp.arange(n_graph)
    total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
    node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)
    return node_graph_idx, n_graph, n_node


def global_graph_aggr(feature, node_graph_idx, n_graph):
    """Aggregate features from nodes to graphs."""
    aggr_feature = jax.ops.segment_sum(feature, node_graph_idx, n_graph)
    return aggr_feature


class ContinuousHead(nn.Module):
    """
    Head module for continuous outputs (chip placement positions).

    Instead of outputting categorical logits for discrete states (like MaxCut spins),
    this outputs mean and log_variance for Gaussian distributions over continuous
    2D positions.

    This replaces the discrete RLHeadModule for chip placement tasks where we need
    to predict continuous (x, y) coordinates.

    @param continuous_dim: Dimension of continuous variables (2 for 2D positions)
    @param value_hidden_dims: Hidden layer sizes for value network
    @param dtype: Data type for computations (float32 or bfloat16)
    """
    continuous_dim: int = 2  # (x, y) positions
    value_hidden_dims: list = None
    dtype: any = jnp.float32

    def setup(self):
        # Default value network architecture if not specified
        if self.value_hidden_dims is None:
            self.value_hidden_dims = [120, 64, 1]

        # Mean network: predicts mean of Gaussian for each dimension
        # Input: embeddings from GNN
        # Output: (batch, continuous_dim) - mean positions
        self.mean_layer = nn.Dense(
            features=self.continuous_dim,
            dtype=self.dtype,
            name="position_mean"
        )

        # Log variance network: predicts log variance of Gaussian
        # We predict log(variance) instead of variance for numerical stability
        # and to ensure variance is always positive (exp(log_var) > 0)
        self.log_var_layer = nn.Dense(
            features=self.continuous_dim,
            dtype=self.dtype,
            name="position_log_var"
        )

        # Value network: estimates V(s) for PPO
        # This predicts the expected return from current state
        # Architecture: embeddings -> hidden layers -> scalar value
        value_layers = []
        for i, hidden_size in enumerate(self.value_hidden_dims):
            value_layers.append(nn.Dense(hidden_size, dtype=self.dtype, name=f"value_dense_{i}"))
        self.value_layers = value_layers

    @partial(flax.linen.jit, static_argnums=0)
    def __call__(self, jraph_graph_list, embeddings, out_dict) -> dict:
        """
        Forward pass for continuous head.

        Args:
            jraph_graph_list: Graph structure information
            embeddings: Node embeddings from GNN (shape: [num_components, 1, embedding_dim])
            out_dict: Dictionary to store outputs

        Returns:
            out_dict with added keys:
                - "position_mean": Mean of Gaussian (shape: [num_components, 1, continuous_dim])
                - "position_log_var": Log variance of Gaussian (shape: [num_components, 1, continuous_dim])
                - "Values": State value estimates (shape: [num_graphs,])
        """
        # embeddings shape: [num_components, 1, embedding_dim]
        # We keep the middle dimension (1) for compatibility with existing code structure

        # Predict mean: MUST be bounded to prevent explosion
        # Without bounds, positions can explode to ±10,000, causing HPWL catastrophe
        # Spread regularization ENCOURAGES spreading but doesn't limit HOW MUCH
        # → Positive feedback loop: spread to ±∞ gives low spread penalty but infinite HPWL!
        # Canvas is [-1, 1], so use tanh to bound to [-1.5, 1.5] with soft boundaries
        position_mean_raw = self.mean_layer(embeddings)  # [num_components, 1, continuous_dim]
        position_mean = 1.5 * jnp.tanh(position_mean_raw)  # Bounded to [-1.5, 1.5]

        # Predict log variance: clip to prevent numerical instability
        # CRITICAL FIX for large datasets: raise variance floor to prevent premature collapse
        # Old: log_var in [-10, 2] → std in [0.0067, 2.718] - too confident for dense placements!
        # New: log_var in [-4, 2] → std in [0.135, 2.718] - maintains exploration longer
        # This prevents the "mean_prob explosion at epoch 300" bug on large datasets
        position_log_var = self.log_var_layer(embeddings)  # [num_components, 1, continuous_dim]
        position_log_var = jnp.clip(position_log_var, -4.0, 2.0)  # Raised floor: -10 → -4

        # Store position predictions
        out_dict["position_mean"] = position_mean
        out_dict["position_log_var"] = position_log_var

        # Compute value function: V(s)
        # Aggregate node embeddings to graph level, then predict scalar value
        node_graph_idx, n_graph, n_node = get_graph_info(jraph_graph_list)

        # Mean pooling of embeddings per graph
        # Normalize by sqrt(n_node) for scale invariance
        value_embeddings = global_graph_aggr(embeddings, node_graph_idx, n_graph) / jnp.sqrt(n_node[..., None, None])

        # Pass through value MLP
        value_out = value_embeddings
        for i, layer in enumerate(self.value_layers[:-1]):
            value_out = layer(value_out)
            value_out = nn.relu(value_out)

        # Final layer: no activation (value can be any real number)
        value_out = self.value_layers[-1](value_out)
        Values = value_out[..., 0, 0]  # [num_graphs,]

        out_dict["Values"] = Values

        return out_dict


class ContinuousHeadChip(nn.Module):
    """
    Alternative continuous head specifically designed for chip placement.

    This version includes additional features for chip placement:
    - Component size awareness (small vs large components may need different variance)
    - Boundary-aware predictions (optionally incorporate canvas bounds)

    Use this if you want more chip-placement-specific features.
    Otherwise, the basic ContinuousHead above is sufficient and more general.

    @param continuous_dim: Dimension of continuous variables (2 for 2D positions)
    @param size_conditioning: Whether to condition on component sizes
    @param value_hidden_dims: Hidden layer sizes for value network
    @param dtype: Data type for computations
    """
    continuous_dim: int = 2
    size_conditioning: bool = True
    value_hidden_dims: list = None
    dtype: any = jnp.float32

    def setup(self):
        if self.value_hidden_dims is None:
            self.value_hidden_dims = [120, 64, 1]

        # If size conditioning, we'll have a size embedding layer
        if self.size_conditioning:
            self.size_embed = nn.Dense(32, dtype=self.dtype, name="size_embedding")

        # Position prediction layers
        self.mean_layer = nn.Dense(
            features=self.continuous_dim,
            dtype=self.dtype,
            name="position_mean"
        )

        self.log_var_layer = nn.Dense(
            features=self.continuous_dim,
            dtype=self.dtype,
            name="position_log_var"
        )

        # Value network
        value_layers = []
        for i, hidden_size in enumerate(self.value_hidden_dims):
            value_layers.append(nn.Dense(hidden_size, dtype=self.dtype, name=f"value_dense_{i}"))
        self.value_layers = value_layers

    @partial(flax.linen.jit, static_argnums=0)
    def __call__(self, jraph_graph_list, embeddings, out_dict, component_sizes=None) -> dict:
        """
        Forward pass for chip-specific continuous head.

        Args:
            jraph_graph_list: Graph structure information
            embeddings: Node embeddings from GNN
            out_dict: Dictionary to store outputs
            component_sizes: Optional component sizes (shape: [num_components, 2]) for x_size, y_size

        Returns:
            out_dict with position predictions and values
        """
        # Optionally incorporate component size information
        if self.size_conditioning and component_sizes is not None:
            # Embed sizes and concatenate with node embeddings
            size_features = self.size_embed(component_sizes)  # [num_components, 32]
            size_features = jnp.expand_dims(size_features, axis=1)  # [num_components, 1, 32]
            embeddings = jnp.concatenate([embeddings, size_features], axis=-1)

        # Predict positions
        # CRITICAL: Bound position_mean to prevent explosion (see ContinuousHead for details)
        position_mean_raw = self.mean_layer(embeddings)
        position_mean = 1.5 * jnp.tanh(position_mean_raw)  # Bounded to [-1.5, 1.5]

        position_log_var = self.log_var_layer(embeddings)
        position_log_var = jnp.clip(position_log_var, -4.0, 2.0)  # Raised floor to prevent variance collapse

        out_dict["position_mean"] = position_mean
        out_dict["position_log_var"] = position_log_var

        # Compute value function
        node_graph_idx, n_graph, n_node = get_graph_info(jraph_graph_list)
        value_embeddings = global_graph_aggr(embeddings, node_graph_idx, n_graph) / jnp.sqrt(n_node[..., None, None])

        value_out = value_embeddings
        for i, layer in enumerate(self.value_layers[:-1]):
            value_out = layer(value_out)
            value_out = nn.relu(value_out)

        value_out = self.value_layers[-1](value_out)
        Values = value_out[..., 0, 0]

        out_dict["Values"] = Values

        return out_dict
