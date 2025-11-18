import jax
import numpy as np
import jax.numpy as jnp
import flax
import flax.linen as nn
from functools import partial

from Networks.Modules import get_GNN_model

class DiffModel(nn.Module):
	"""
	Policy Network
	"""
	n_features_list_prob: np.ndarray

	n_features_list_nodes: np.ndarray
	n_features_list_edges: np.ndarray
	n_features_list_messages: np.ndarray

	n_features_list_encode: np.ndarray
	n_features_list_decode: np.ndarray

	n_diffusion_steps: int
	n_message_passes: int
	edge_updates: bool
	problem_type: str

	time_encoding: str
	n_diff_steps: int
	embedding_dim: int = 32
	message_passing_weight_tied: bool = True
	linear_message_passing: bool = True
	n_bernoulli_features: int = 1
	mean_aggr: bool = False
	EncoderModel: str = "normal"
	n_random_node_features: int = 5
	train_mode: str = "REINFORCE"
	graph_norm: bool = False
	bfloat16: bool = False
	dataset_name: str = "None"
	continuous_dim: int = 0  # If > 0, use continuous mode (e.g., 2 for 2D positions)



	def setup(self):
		if(self.bfloat16 == False):
			dtype = jnp.float32
		else:
			dtype = jnp.bfloat16

		GNNModel, HeadModel = get_GNN_model(self.EncoderModel, self.train_mode)
		if(self.EncoderModel != "UNet"):
			self.encode_process_decode = GNNModel(dtype = dtype, n_features_list_nodes=self.n_features_list_nodes,
																 n_features_list_edges=self.n_features_list_edges,
																 n_features_list_messages=self.n_features_list_messages,
																 n_features_list_encode=self.n_features_list_encode,
																 n_features_list_decode=self.n_features_list_decode,
																 edge_updates=self.edge_updates,
																 n_message_passes=self.n_message_passes,
																 weight_tied=self.message_passing_weight_tied,
																 linear_message_passing=self.linear_message_passing,
																 mean_aggr = self.mean_aggr,
																 graph_norm = self.graph_norm)
		else:
			import re
			def extract_integer(input_string):
				# Use a regular expression to find digits at the end of the string
				match = re.search(r'\d+$', input_string)
				if match:
					return int(match.group())
				else:
					return None
			size = extract_integer(self.dataset_name)

			self.encode_process_decode = GNNModel(size = size, features=self.n_features_list_nodes[0],
																 n_layers=self.n_message_passes
																 )

		# Pass continuous_dim to HeadModel for continuous problems (e.g., ChipPlacement)
		self.HeadModel = HeadModel(n_features_list_prob=self.n_features_list_prob, dtype = dtype, continuous_dim=self.continuous_dim)

		self.__vmap_get_log_probs = jax.vmap(self.__get_log_prob, in_axes=(0, None, None), out_axes=(0))
		self.vamp_get_sinusoidal_positional_encoding = jax.vmap(get_sinusoidal_positional_encoding, in_axes=(0, None, None))
		### TODO random node feature key is different during eval and sample, force them to be the same?

	@flax.linen.jit
	def __call__(self, jraph_graph_list, X_prev, rand_node_features, t_idx_per_node, key):
		X_prev = self._add_random_nodes_and_time_index(X_prev, rand_node_features, t_idx_per_node)
		embeddings = self.encode_process_decode(jraph_graph_list, X_prev)

		bernoulli_embeddings = jnp.repeat(embeddings[:, jnp.newaxis, :], 1, axis = -2)
		embeddings = bernoulli_embeddings

		out_dict = {}
		out_dict = self.HeadModel(jraph_graph_list, embeddings, out_dict)

		out_dict["rand_node_features"] = rand_node_features
		return out_dict, key

	#@partial(flax.linen.jit, static_argnums=0)
	def get_graph_info(self, jraph_graph_list):
		first_graph = jraph_graph_list["graphs"][0]
		nodes = first_graph.nodes
		n_node = first_graph.n_node
		n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
		graph_idx = jnp.arange(n_graph)
		total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
		node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)
		return node_graph_idx, n_graph, n_node

	@partial(flax.linen.jit, static_argnums=0)
	def reinit_rand_nodes(self, X_t,  key):
		key, subkey = jax.random.split(key)
		rand_nodes = jax.random.uniform(subkey, shape=(X_t.shape[0], self.n_random_node_features))

		return rand_nodes, key

	@partial(flax.linen.jit, static_argnums=(0,))
	def _add_random_nodes_and_time_index(self, X_t, rand_nodes, t_idx_per_node):
		# Time encoding (same for discrete and continuous)
		if(self.time_encoding == "one_hot"):
			X_embed = jax.nn.one_hot(jnp.squeeze(t_idx_per_node, axis = -1), num_classes=self.n_diffusion_steps)
		else:
			X_embed = self.vamp_get_sinusoidal_positional_encoding(jnp.squeeze(t_idx_per_node, axis = -1), self.embedding_dim, self.n_diff_steps)

		# State encoding: different for discrete vs continuous
		if self.continuous_dim > 0:
			# Continuous mode: X_t is already continuous positions [num_components, continuous_dim]
			# Ensure shape is [num_components, continuous_dim]
			if len(X_t.shape) == 3:  # [num_components, 1, continuous_dim]
				X_state = X_t[:, 0, :]
			elif len(X_t.shape) == 2:  # [num_components, continuous_dim]
				X_state = X_t
			else:
				raise ValueError(f"Unexpected X_t shape in continuous mode: {X_t.shape}")
		else:
			# Discrete mode: one-hot encode discrete states
			X_state = jax.nn.one_hot(X_t[...,0], num_classes=self.n_bernoulli_features)

		X_input = jnp.concatenate([X_state, X_embed, rand_nodes], axis=-1)
		return X_input

	@partial(flax.linen.jit, static_argnums=0)
	def make_one_step(self,params ,jraph_graph_list, X_prev, t_idx_per_node, key):
		rand_nodes, key = self.reinit_rand_nodes(X_prev, key)

		out_dict, key = self.apply(params, jraph_graph_list, X_prev, rand_nodes, t_idx_per_node, key)

		node_graph_idx, n_graph, n_node = self.get_graph_info(jraph_graph_list)

		if self.continuous_dim > 0:
			# Continuous mode
			output_params = out_dict  # Contains "position_mean" and "position_log_var"
			X_next, position_log_probs, key = self.sample_from_model(output_params, key)

			# Aggregate log probs to graph level
			# position_log_probs shape: [num_components, 1] or [num_components,]
			if len(position_log_probs.shape) > 1:
				position_log_probs_flat = position_log_probs[..., 0]
			else:
				position_log_probs_flat = position_log_probs

			state_log_probs = self.__get_log_prob(position_log_probs_flat, node_graph_idx, n_graph)
			graph_log_prob = jax.lax.stop_gradient(jnp.exp((state_log_probs / n_node)[:-1]))

			out_dict["X_next"] = X_next
			out_dict["position_log_probs"] = position_log_probs
			out_dict["state_log_probs"] = state_log_probs
			out_dict["graph_log_prob"] = graph_log_prob
		else:
			# Discrete mode
			spin_logits = out_dict["spin_logits"]
			X_next, spin_log_probs, key = self.sample_from_model(spin_logits, key)

			graph_log_prob = jax.lax.stop_gradient(jnp.exp((self.__get_log_prob(spin_log_probs[...,0], node_graph_idx, n_graph)/(n_node))[:-1]))
			out_dict["X_next"] = X_next
			out_dict["spin_log_probs"] = spin_log_probs
			out_dict["state_log_probs"] = self.__get_log_prob(spin_log_probs[...,0], node_graph_idx, n_graph)
			out_dict["graph_log_prob"] = graph_log_prob

		return out_dict, key

	@partial(flax.linen.jit, static_argnums=0)
	def unbiased_last_step(self,params ,jraph_graph_list, X_prev, t_idx, key, eps = 0.01):
		rand_nodes, key = self.reinit_rand_nodes(X_prev, key)
		out_dict, key = self.apply(params, jraph_graph_list, rand_nodes, X_prev, t_idx, key)

		spin_logits = out_dict["spin_logits"]
		j_graphs = jraph_graph_list["graphs"][0]
		key, subkey = jax.random.split(key)

		sampled_p = jax.random.uniform(key, shape =  (j_graphs.n_node.shape[0],))

		nodes = j_graphs.nodes
		n_node = j_graphs.n_node
		total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
		graph_sampled_p = jnp.repeat(sampled_p, n_node, axis=0, total_repeat_length=total_nodes)
		graph_sampled_p = graph_sampled_p[:, None]

		X_next_model, spin_log_probs_model, key = self.sample_from_model(spin_logits, key)
		X_next_uniform, one_hot_state, log_p_uniform_density, key = self.sample_prior(j_graphs, spin_logits.shape[1],  key)
		X_next_uniform = X_next_uniform[...,0]
		log_p_uniform = jnp.sum(log_p_uniform_density * one_hot_state, axis=-1)[...,0]

		X_next = jnp.where(graph_sampled_p < eps, X_next_uniform, X_next_model)

		concat_spin_log_probs = jnp.concatenate([spin_log_probs_model[None,...], log_p_uniform[None, ...]], axis = 0)
		weights =  jnp.concatenate([(1-eps)*jnp.ones_like(spin_log_probs_model)[None,...], eps*jnp.ones_like(spin_log_probs_model)[None, ...]], axis = 0)
		spin_log_probs = jax.scipy.special.logsumexp(concat_spin_log_probs, axis = 0, b = weights)

		node_graph_idx, n_graph, n_node = self.get_graph_info(jraph_graph_list)

		graph_log_prob = jax.lax.stop_gradient(jnp.exp((self.__get_log_prob(jnp.sum(spin_log_probs, axis = -1), node_graph_idx, n_graph)/(n_node))[:-1]))
		return X_next, spin_log_probs, spin_logits, graph_log_prob, key

	@partial(flax.linen.jit, static_argnums=0)
	def sample_from_model(self, output_params, key):
		"""
		Sample from model output.

		Args:
			output_params: For discrete: spin_logits [num_nodes, 1, num_classes]
			               For continuous: tuple of (position_mean, position_log_var)
			key: JAX random key

		Returns:
			X_next: Sampled state
			log_probs: Log probability of sample
			key: Updated random key
		"""
		key, subkey = jax.random.split(key)

		if self.continuous_dim > 0:
			# Continuous mode: sample from Gaussian using reparameterization trick
			# output_params should be dict with "position_mean" and "position_log_var"
			if isinstance(output_params, dict):
				position_mean = output_params["position_mean"]  # [num_components, 1, continuous_dim]
				position_log_var = output_params["position_log_var"]
			else:
				# Assume tuple (mean, log_var)
				position_mean, position_log_var = output_params

			# Reparameterization trick: X = mean + std * epsilon
			epsilon = jax.random.normal(subkey, shape=position_mean.shape)
			std = jnp.exp(0.5 * position_log_var)
			X_next = position_mean + std * epsilon

			# Compute log probability: log N(X | mean, var)
			# log p = -0.5 * [epsilon^2 + log_var + log(2*pi)]
			log_prob_per_dim = -0.5 * (epsilon**2 + position_log_var + jnp.log(2 * jnp.pi))
			log_probs = jnp.sum(log_prob_per_dim, axis=-1)  # Sum over continuous_dim

			# Remove middle dimension if present: [num_components, 1] -> [num_components,]
			if len(X_next.shape) == 3:
				X_next = X_next[:, 0, :]  # [num_components, continuous_dim]

		else:
			# Discrete mode: sample from categorical distribution
			spin_logits = output_params
			X_next = jax.random.categorical(key=subkey,
											   logits=spin_logits,
											   axis=-1,
											   shape=spin_logits.shape[:-1])

			one_hot_state = jax.nn.one_hot(X_next, num_classes=self.n_bernoulli_features)
			log_probs = jnp.sum(spin_logits * one_hot_state, axis=-1)

		return X_next, log_probs, key

	@partial(flax.linen.jit, static_argnums=0)
	def calc_log_q(self, params, jraph_graph_list, X_prev, rand_nodes, X_next, t_idx_per_node, key):
		"""
		Calculate log probability of X_next under the model's predicted distribution.

		Args:
			params: Model parameters
			jraph_graph_list: Graph structure
			X_prev: Previous state
			rand_nodes: Random node features
			X_next: Next state (for discrete: integer labels, for continuous: positions)
			t_idx_per_node: Time index per node
			key: Random key

		Returns:
			out_dict: Dictionary with log probabilities
			key: Updated random key
		"""
		out_dict, key = self.apply(params, jraph_graph_list, X_prev, rand_nodes, t_idx_per_node, key)
		node_graph_idx, n_graph, n_node = self.get_graph_info(jraph_graph_list)

		if self.continuous_dim > 0:
			# Continuous mode: compute log N(X_next | mean, var)
			position_mean = out_dict["position_mean"]  # [num_components, 1, continuous_dim]
			position_log_var = out_dict["position_log_var"]

			# Ensure X_next has correct shape
			if len(X_next.shape) == 2:  # [num_components, continuous_dim]
				X_next_expanded = X_next[:, None, :]  # [num_components, 1, continuous_dim]
			elif len(X_next.shape) == 3:
				X_next_expanded = X_next
			else:
				raise ValueError(f"Unexpected X_next shape in continuous mode: {X_next.shape}")

			# Compute log probability: log N(X_next | mean, var)
			# log p = -0.5 * [(X - mean)^2 / var + log(var) + log(2*pi)]
			diff = X_next_expanded - position_mean
			log_prob_per_dim = -0.5 * (diff**2 / jnp.exp(position_log_var) + position_log_var + jnp.log(2 * jnp.pi))
			position_log_probs = jnp.sum(log_prob_per_dim, axis=-1)  # Sum over continuous_dim -> [num_components, 1]

			X_next_log_prob = self.__get_log_prob(position_log_probs[:, 0], node_graph_idx, n_graph)

			out_dict["state_log_probs"] = X_next_log_prob
			out_dict["position_log_probs"] = position_log_probs
		else:
			# Discrete mode
			spin_logits = out_dict["spin_logits"]

			one_hot_state = jax.nn.one_hot(X_next, num_classes=self.n_bernoulli_features)
			spin_log_probs = jnp.sum(spin_logits * one_hot_state, axis=-1)
			X_next_log_prob = self.__get_log_prob(spin_log_probs[...,0], node_graph_idx, n_graph)

			out_dict["state_log_probs"] = X_next_log_prob
			out_dict["spin_log_probs"] = spin_log_probs

		return out_dict, key

	@partial(flax.linen.jit, static_argnums=0)
	def calc_log_q_T(self, j_graph, X_T):
		'''
		Calculate log probability of X_T under the prior distribution.

		:param j_graph: Graph structure
		:param X_T: For discrete: shape = (batched_graph_nodes, n_states, 1)
		            For continuous: shape = (batched_graph_nodes, n_states) or (batched_graph_nodes, n_states, continuous_dim)
		:return: log_p_X_T: log probability per graph
		'''
		nodes = j_graph.nodes
		n_node = j_graph.n_node
		n_graph = j_graph.n_node.shape[0]
		graph_idx = jnp.arange(n_graph)
		total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
		node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)

		if self.continuous_dim > 0:
			# Continuous mode: compute log N(X_T | 0, I)
			# Prior is N(0, I), so log p(X) = -0.5 * (X^2 + log(2*pi))
			# X_T shape: (batched_graph_nodes, n_states) or (batched_graph_nodes, n_states, continuous_dim)

			if len(X_T.shape) == 2:  # (batched_graph_nodes, continuous_dim)
				X_values = X_T[:, None, :]  # Add n_states dimension: (batched_graph_nodes, 1, continuous_dim)
			elif len(X_T.shape) == 3:  # (batched_graph_nodes, n_states, continuous_dim)
				X_values = X_T  # Use all states
			else:
				raise ValueError(f"Unexpected X_T shape in continuous mode: {X_T.shape}")

			# Compute log probability per dimension, then sum
			log_p_per_dim = -0.5 * (X_values**2 + jnp.log(2 * jnp.pi))
			log_p_X_T_per_node = jnp.sum(log_p_per_dim, axis=-1)  # Sum over continuous_dim
		else:
			# Discrete mode: use one-hot encoding
			shape = X_T.shape[0:-1]
			log_p_uniform = self._get_prior(shape)

			one_hot_state = jax.nn.one_hot(X_T[...,-1], num_classes=self.n_bernoulli_features)
			log_p_X_T_per_node = jnp.sum(log_p_uniform * one_hot_state, axis=-1)

		log_p_X_T = self.__get_log_prob(log_p_X_T_per_node, node_graph_idx, n_graph)

		return log_p_X_T

	@partial(flax.linen.jit, static_argnums=(0,2))
	def sample_prior(self, j_graph, N_basis_states, key):
		"""
		Sample from prior distribution.

		For discrete: sample from uniform categorical
		For continuous: sample from standard Gaussian N(0, I)
		"""
		nodes = j_graph.nodes
		num_nodes = nodes.shape[0]

		key, subkey = jax.random.split(key)

		if self.continuous_dim > 0:
			# Continuous mode: sample from N(0, I)
			shape = (num_nodes, N_basis_states)
			prior_mean, prior_log_var = self._get_prior(shape)

			# Sample using reparameterization
			epsilon = jax.random.normal(subkey, shape=prior_mean.shape)
			X_prev = prior_mean + jnp.exp(0.5 * prior_log_var) * epsilon

			# For continuous, we don't have one_hot_state
			# Return X_prev as-is, and prior params
			one_hot_state = None  # Not applicable for continuous
			prior_params = (prior_mean, prior_log_var)
			return X_prev, one_hot_state, prior_params, key
		else:
			# Discrete mode: sample from uniform categorical
			shape = (num_nodes, N_basis_states, 1)
			log_p_uniform = self._get_prior(shape)

			X_prev = jax.random.categorical(key=subkey,
											logits=log_p_uniform,
											axis=-1,
											shape=log_p_uniform.shape[:-1])

			one_hot_state = jax.nn.one_hot(X_prev, num_classes=self.n_bernoulli_features)
			return X_prev, one_hot_state, log_p_uniform, key

	@partial(flax.linen.jit, static_argnums=(0,2))
	def sample_prior_w_probs(self, j_graph, N_basis_states, key):
		"""
		Sample from prior and compute log probabilities.

		Returns:
			For discrete: X_T, log_p_X_T, one_hot_state, log_p_uniform, key
			For continuous: X_T, log_p_X_T, None, prior_params, key
		"""
		X_T, one_hot_state, prior_params, key = self.sample_prior(j_graph, N_basis_states, key)
		log_p_X_T = self.calc_log_q_T(j_graph, X_T)
		return X_T, log_p_X_T, one_hot_state, prior_params, key

	@partial(flax.linen.jit, static_argnums=(0,1))
	def _get_prior(self, shape):
		"""
		Get prior distribution.

		For discrete: uniform categorical distribution
		For continuous: standard Gaussian N(0, I)

		Args:
			shape: Base shape (num_nodes, N_basis_states, 1) for discrete
			       or (num_components, N_basis_states) for continuous

		Returns:
			For discrete: log probabilities of uniform categorical
			For continuous: tuple of (mean, log_var) for standard Gaussian
		"""
		if self.continuous_dim > 0:
			# Continuous prior: N(0, I)
			# Mean = 0, log_var = log(1) = 0
			mean = jnp.zeros(shape + (self.continuous_dim,))
			log_var = jnp.zeros(shape + (self.continuous_dim,))
			return mean, log_var
		else:
			# Discrete prior: uniform categorical
			log_p_uniform = jnp.log(1./self.n_bernoulli_features * jnp.ones(shape +  (self.n_bernoulli_features, )))
			return log_p_uniform

	#@partial(flax.linen.jit, static_argnums=(0,-1))
	def __get_log_prob(self, spin_log_probs, node_graph_idx, n_graph):
		log_probs = self.__global_graph_aggr(spin_log_probs, node_graph_idx, n_graph)
		return log_probs

	#@partial(flax.linen.jit, static_argnums=(0,-1))
	def __global_graph_aggr(self, feature, node_graph_idx, n_graph):
		aggr_feature = jax.ops.segment_sum(feature, node_graph_idx, n_graph)
		return aggr_feature


def get_sinusoidal_positional_encoding(timestep, embedding_dim, max_position):
	"""
    Create a sinusoidal positional encoding as described in the
    "Attention is All You Need" paper.

    Args:
        timestep (int): The current time step.
        embedding_dim (int): The dimensionality of the encoding.

    Returns:
        A 1D tensor of shape (embedding_dim,) representing the
        positional encoding for the given timestep.
    """
	position = timestep
	div_term = jnp.exp(np.arange(0, embedding_dim, 2) * (-jnp.log(max_position) / embedding_dim))
	return jnp.concatenate([jnp.sin(position * div_term), jnp.cos(position * div_term)], axis=-1)
