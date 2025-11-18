from .BaseNoise import BaseNoiseDistr
import jax.numpy as jnp
import jax
from functools import partial

class GaussianNoiseDistr(BaseNoiseDistr):
    """
    Gaussian noise distribution for continuous diffusion.

    For continuous variables (e.g., chip placement positions), we use Gaussian diffusion:
    - Forward process: X_t = sqrt(1 - beta_t) * X_{t-1} + sqrt(beta_t) * epsilon
    - Reverse process: Model predicts parameters of Gaussian distribution

    This replaces BernoulliNoiseDistr which was used for discrete binary variables (MaxCut).
    """

    def __init__(self, config):
        super().__init__(config)

        # Store dimension of continuous variables (e.g., 2 for 2D positions)
        self.continuous_dim = config.get("continuous_dim", 2)

        # Precompute alpha values for efficiency
        # alpha_t = 1 - beta_t
        self.alpha_arr = 1.0 - self.beta_arr

        # Precompute cumulative products for direct sampling from X_0 to X_t
        # alpha_bar_t = prod_{i=1}^{t} alpha_i
        self.alpha_bar_arr = jnp.cumprod(self.alpha_arr)

        print("GaussianNoise initialized")
        print(f"  Continuous dim: {self.continuous_dim}")
        print(f"  Alpha values: {self.alpha_arr[:5]}... (showing first 5)")
        print(f"  Alpha_bar values: {self.alpha_bar_arr[:5]}... (showing first 5)")
        print("______________")

    def combine_losses(self, L_entropy, L_noise, L_energy, T):
        """
        Combine losses for PPO training.

        Args:
            L_entropy: Entropy loss (encourages exploration)
            L_noise: Noise matching loss (KL divergence to forward process)
            L_energy: Energy-based reward (HPWL + constraints)
            T: Temperature parameter

        Returns:
            Combined loss
        """
        return -T * L_entropy + L_noise + L_energy

    def calculate_noise_distr_reward(self, noise_distr_step, entropy_reward):
        """
        Calculate reward for noise distribution matching.

        Args:
            noise_distr_step: Negative log probability of forward transition
            entropy_reward: Entropy of predicted distribution

        Returns:
            Reward (higher is better, so negate the loss)
        """
        return -(noise_distr_step - entropy_reward)

    @partial(jax.jit, static_argnums=(0,))
    def get_log_p_T_0(self, jraph_graph, X_prev, X_next, t_idx, T):
        """
        Compute log probability of transition X_prev -> X_next at timestep t.

        For Gaussian diffusion:
        p(X_t | X_{t-1}) = N(X_t; sqrt(1 - beta_t) * X_{t-1}, beta_t * I)

        Args:
            jraph_graph: Graph structure
            X_prev: Previous state (shape: [num_nodes * num_components, continuous_dim])
            X_next: Next state (shape: [num_nodes * num_components, continuous_dim])
            t_idx: Timestep index
            T: Temperature (not used directly here)

        Returns:
            Log probability per graph (shape: [n_graphs,])
        """
        nodes = jraph_graph.nodes
        n_node = jraph_graph.n_node
        n_graph = jraph_graph.n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        node_gr_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)

        # Compute log probability per component (each component is a node with continuous_dim values)
        log_p_per_component = self.get_log_p_T_0_per_component(X_prev, X_next, t_idx)

        # Aggregate to graph level
        log_p_per_graph = jax.ops.segment_sum(log_p_per_component, node_gr_idx, n_graph)

        return log_p_per_graph

    @partial(jax.jit, static_argnums=(0,))
    def get_log_p_T_0_per_component(self, X_prev, X_next, t_idx):
        """
        Compute log probability of Gaussian transition for each component.

        p(X_t | X_{t-1}) = N(X_t; mu, sigma^2 * I)
        where:
            mu = sqrt(1 - beta_t) * X_{t-1}
            sigma^2 = beta_t

        log p = -0.5 * sum_d [(X_t^d - mu^d)^2 / sigma^2 + log(2*pi*sigma^2)]

        Args:
            X_prev: Previous positions (shape: [num_components, continuous_dim])
            X_next: Next positions (shape: [num_components, continuous_dim])
            t_idx: Timestep index

        Returns:
            Log probability per component (shape: [num_components,])
        """
        beta_t = self.beta_arr[t_idx]
        alpha_t = self.alpha_arr[t_idx]

        # Mean: sqrt(alpha_t) * X_prev = sqrt(1 - beta_t) * X_prev
        mean = jnp.sqrt(alpha_t) * X_prev

        # Variance: beta_t
        var = beta_t
        std = jnp.sqrt(var)

        # Compute log probability for multivariate Gaussian (independent dimensions)
        # log p = -0.5 * sum_d [(x - mu)^2 / sigma^2 + log(2*pi*sigma^2)]
        diff = X_next - mean
        log_prob_per_dim = -0.5 * (diff**2 / var + jnp.log(2 * jnp.pi * var))

        # Sum over continuous dimensions (e.g., x and y for 2D positions)
        log_prob_per_component = jnp.sum(log_prob_per_dim, axis=-1)

        return log_prob_per_component

    @partial(jax.jit, static_argnums=(0,))
    def sample_forward_diff_process(self, X_t_m1, t_idx, key):
        """
        Sample from forward diffusion process: X_t ~ p(X_t | X_{t-1}).

        X_t = sqrt(1 - beta_t) * X_{t-1} + sqrt(beta_t) * epsilon
        where epsilon ~ N(0, I)

        Args:
            X_t_m1: Previous state (shape: [num_components, continuous_dim])
            t_idx: Timestep index
            key: JAX random key

        Returns:
            X_t: Next state (shape: [num_components, continuous_dim])
            log_probs: Log probability of this sample (shape: [num_components,])
            key: Updated JAX random key
        """
        beta_t = self.beta_arr[t_idx]
        alpha_t = self.alpha_arr[t_idx]

        # Mean and std for forward process
        mean = jnp.sqrt(alpha_t) * X_t_m1
        std = jnp.sqrt(beta_t)

        # Sample noise
        key, subkey = jax.random.split(key)
        epsilon = jax.random.normal(subkey, shape=X_t_m1.shape)

        # Sample next state
        X_t = mean + std * epsilon

        # Compute log probability of this sample
        diff = X_t - mean
        log_prob_per_dim = -0.5 * (diff**2 / (std**2) + jnp.log(2 * jnp.pi * std**2))
        log_probs = jnp.sum(log_prob_per_dim, axis=-1)  # Sum over dimensions

        return X_t, log_probs, key

    @partial(jax.jit, static_argnums=(0,))
    def calc_noise_loss(self, jraph_graph, mean_prev, log_var_prev, mean_next, log_var_next, X_prev, log_p_prev_per_component, model_step_idx, node_gr_idx, T):
        """
        Calculate noise matching loss for continuous diffusion.

        This measures KL divergence between:
        - Forward process: p(X_t | X_{t-1}) = N(sqrt(alpha_t) * X_{t-1}, beta_t * I)
        - Reverse model: q(X_t | X_{t-1}) = N(mean_next, exp(log_var_next) * I)

        Args:
            jraph_graph: Graph structure
            mean_prev: Mean from previous step (not used in forward, but kept for interface consistency)
            log_var_prev: Log variance from previous step (not used)
            mean_next: Predicted mean for next step
            log_var_next: Predicted log variance for next step
            X_prev: Current state
            log_p_prev_per_component: Log prob from previous sampling
            model_step_idx: Current timestep
            node_gr_idx: Node to graph index mapping
            T: Temperature

        Returns:
            noise_loss_per_graph: Noise loss aggregated per graph
            total_log_p_prev: Total log probability from previous step
        """
        beta_t = self.beta_arr[model_step_idx]
        alpha_t = self.alpha_arr[model_step_idx]

        # Ground truth forward process parameters
        true_mean = jnp.sqrt(alpha_t) * X_prev
        true_var = beta_t

        # Predicted parameters
        pred_mean = mean_next
        pred_var = jnp.exp(log_var_next)

        # KL divergence between two Gaussians: KL(N(mu1, sig1^2) || N(mu2, sig2^2))
        # KL = 0.5 * [log(sig2^2/sig1^2) + (sig1^2 + (mu1-mu2)^2)/sig2^2 - 1]
        # Here: true = p (forward), pred = q (model)
        # We want KL(p || q) to match forward process with model

        diff_mean = true_mean - pred_mean

        # Per dimension KL divergence
        kl_per_dim = 0.5 * (
            jnp.log(pred_var / true_var) +
            (true_var + diff_mean**2) / pred_var -
            1.0
        )

        # Sum over dimensions to get KL per component
        kl_per_component = jnp.sum(kl_per_dim, axis=-1)

        # Aggregate to graph level
        n_graph = jraph_graph.n_node.shape[0]
        noise_per_graph = jax.ops.segment_sum(kl_per_component, node_gr_idx, n_graph)

        # Return with temperature scaling
        return T * noise_per_graph, jnp.sum(log_p_prev_per_component, axis=0)

    @partial(jax.jit, static_argnums=(0,))
    def calc_noise_step_relaxed(self, jraph_graph, mean_prev, log_var_prev, mean_next, log_var_next, X_prev, gamma_t, node_gr_idx):
        """
        Calculate relaxed noise step (used during training with soft samples).

        Similar to calc_noise_loss but with custom gamma_t parameter.

        Args:
            jraph_graph: Graph structure
            mean_prev: Mean from previous step
            log_var_prev: Log variance from previous step
            mean_next: Predicted mean for next step
            log_var_next: Predicted log variance for next step
            X_prev: Current state
            gamma_t: Beta value for this step
            node_gr_idx: Node to graph index mapping

        Returns:
            noise_per_graph: Noise loss per graph
        """
        alpha_t = 1.0 - gamma_t

        # Ground truth forward process parameters
        true_mean = jnp.sqrt(alpha_t) * X_prev
        true_var = gamma_t

        # Predicted parameters
        pred_mean = mean_next
        pred_var = jnp.exp(log_var_next)

        # KL divergence
        diff_mean = true_mean - pred_mean
        kl_per_dim = 0.5 * (
            jnp.log(pred_var / true_var) +
            (true_var + diff_mean**2) / pred_var -
            1.0
        )

        kl_per_component = jnp.sum(kl_per_dim, axis=-1)

        # Aggregate to graph level
        n_graph = jraph_graph.n_node.shape[0]
        noise_per_graph = jax.ops.segment_sum(kl_per_component, node_gr_idx, n_graph)

        return noise_per_graph

    @partial(jax.jit, static_argnums=(0,))
    def calc_noise_step(self, jraph_graph, X_prev, X_next, model_step_idx, node_gr_idx, T, noise_rewards_arr):
        """
        Calculate noise reward for a single step in the diffusion process.

        This computes the negative log probability of the forward transition and
        accumulates it into the noise rewards array.

        Args:
            jraph_graph: Graph structure
            X_prev: Previous state
            X_next: Next state (sampled from forward process)
            model_step_idx: Current timestep index
            node_gr_idx: Node to graph index mapping
            T: Temperature
            noise_rewards_arr: Array to accumulate noise rewards

        Returns:
            Updated noise_rewards_arr
        """
        beta_t = self.beta_arr[model_step_idx]
        alpha_t = self.alpha_arr[model_step_idx]
        reward_idx = model_step_idx

        # Forward process parameters
        mean = jnp.sqrt(alpha_t) * X_prev
        var = beta_t

        # Compute log probability of X_next given X_prev
        diff = X_next - mean
        log_prob_per_dim = -0.5 * (diff**2 / var + jnp.log(2 * jnp.pi * var))
        log_prob_per_component = jnp.sum(log_prob_per_dim, axis=-1)

        # Aggregate to graph level
        n_graph = jraph_graph.n_node.shape[0]
        log_prob_per_graph = jax.ops.segment_sum(log_prob_per_component, node_gr_idx, n_graph)

        # Noise step value (negative log prob, weighted by temperature)
        noise_step_value = -T * log_prob_per_graph

        # Accumulate into rewards array
        noise_rewards_arr = noise_rewards_arr.at[reward_idx].set(
            noise_rewards_arr[reward_idx] - noise_step_value
        )

        return noise_rewards_arr

    def __get_log_prob(self, log_probs_per_component, node_graph_idx, n_graph):
        """
        Aggregate log probabilities from component level to graph level.

        Args:
            log_probs_per_component: Log probs per component
            node_graph_idx: Mapping from components to graphs
            n_graph: Number of graphs

        Returns:
            Log probabilities per graph
        """
        log_probs = jax.ops.segment_sum(log_probs_per_component, node_graph_idx, n_graph)
        return log_probs
