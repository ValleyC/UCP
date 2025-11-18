"""
Configuration for Continuous SDDS Chip Placement Training

This config enables unsupervised chip placement optimization using:
- Gaussian diffusion (continuous positions)
- ContinuousHead (outputs mean/log_var)
- ChipPlacementEnergy (HPWL + constraints)
- PPO training with GAE

Usage:
    python argparse_ray_main.py --EnergyFunction ChipPlacement \
                                  --dataset Chip_20_components \
                                  --train_mode PPO \
                                  --graph_mode normal \
                                  --noise_potential gaussian \
                                  --GPUs 0
"""

# Base configuration for chip placement
CHIP_PLACEMENT_CONFIG = {
    # ===== Problem Settings =====
    "mode": "Diffusion",
    "dataset_name": "Chip_20_components",  # Will generate 20-50 components per instance
    "problem_name": "ChipPlacement",       # Must match EnergyFunction registry

    # ===== Continuous Mode Settings =====
    "continuous_dim": 2,                   # 2D positions (x, y)
    "n_bernoulli_features": 2,             # Set to continuous_dim for compatibility

    # ===== Diffusion Settings =====
    "noise_potential": "gaussian",         # GaussianNoise for continuous
    "n_diffusion_steps": 1000,             # Number of diffusion steps T (paper: 1000)
    "beta_factor": 1.0,                    # Noise schedule factor
    "diff_schedule": "cosine",             # "cosine" schedule (paper specification)
    "time_encoding": "sinusoidal",         # Better than one_hot for many steps

    # ===== Training Settings =====
    "train_mode": "PPO",                   # Must be PPO for RL-based training
    "lr": 3e-4,                            # Learning rate (paper: 3e-4)
    "lr_schedule": "cosine",               # Learning rate schedule
    "stop_epochs": 1000,                   # Maximum training epochs
    "batch_size": 32,                      # Number of graphs per batch (paper: 32)
    "N_basis_states": 20,                  # Number of parallel states per graph (paper: 20)
    "n_test_basis_states": 20,             # More states for evaluation

    # ===== PPO Hyperparameters =====
    "inner_loop_steps": 4,                 # Number of PPO update iterations (paper: 4 epochs)
    "minib_diff_steps": 10,                # Minibatch size in diffusion steps
    "minib_basis_states": 8,               # Minibatch size in basis states (paper: 8)
    "TD_k": 3,                             # GAE parameter (λ decay)
    "clip_value": 0.2,                     # PPO clip epsilon (paper: 0.2)
    "value_weighting": 0.5,                # Value loss coefficient c_value (paper: 0.5)
    "mov_average": 0.0009,                 # Moving average for normalization
    "grad_clip": True,                     # Gradient clipping (paper: max norm 0.5)

    # ===== Network Architecture =====
    "graph_mode": "normal",                # Use EncodeProcessDecode GNN
    "n_hidden_neurons": 64,                # Hidden dimension
    "n_features_list_prob": [64, 64],      # ContinuousHead will use this
    "n_features_list_nodes": [64, 64, 64], # Node update layers
    "n_features_list_edges": [64, 64],     # Edge update layers
    "n_features_list_messages": [64, 64],  # Message passing layers
    "n_features_list_encode": [64, 64],    # Encoder layers
    "n_features_list_decode": [64, 64],    # Decoder layers
    "n_message_passes": 5,                 # Number of message passing steps
    "message_passing_weight_tied": False,  # Independent weights per layer
    "linear_message_passing": True,        # Use linear layers
    "edge_updates": False,                 # No edge feature updates
    "mean_aggr": True,                     # Mean aggregation in message passing
    "graph_norm": True,                    # Graph normalization

    # ===== Input Features =====
    "random_node_features": True,          # Random node features for exploration
    "n_random_node_features": 5,           # Number of random features
    "time_conditioning": True,             # Condition on diffusion timestep

    # ===== Energy Function Settings =====
    "overlap_weight": 2.0,            # Weight λ_overlap (paper: 2.0)
    "boundary_weight": 1.0,           # Weight λ_bound (paper: 1.0)
    "overlap_threshold": 0.1,         # Normalization threshold for overlap
    "boundary_threshold": 0.1,        # Normalization threshold for boundary

    "canvas_width": 2.0,              # Canvas width (x: [-1, 1])
    "canvas_height": 2.0,             # Canvas height (y: [-1, 1])
    "canvas_x_min": -1.0,             # Canvas x minimum
    "canvas_y_min": -1.0,             # Canvas y minimum

    # ===== Data Settings =====
    "seed": 123,                           # Random seed
    "relaxed": True,                       # Use relaxed (continuous) states
    "jit": True,                           # JIT compilation

    # ===== Annealing Settings (Optional) =====
    # For continuous chip placement, we typically don't use temperature annealing
    # But these are kept for compatibility
    "T_max": 0.0,                          # Maximum temperature (0 for no annealing)
    "T_target": 0.0,                       # Target temperature
    "N_warmup": 0,                         # Warmup epochs
    "N_anneal": 0,                         # Annealing epochs
    "N_equil": 0,                          # Equilibration epochs
    "AnnealSchedule": "linear",            # Annealing schedule

    # ===== Advanced Settings =====
    "loss_alpha": 0.0,                     # KL weighting (0 for pure RL)
    "proj_method": "None",                 # Projection method ("None", "CE", "feasible")
    "sampling_temp": 0.0,                  # Sampling temperature
    "n_sampling_rounds": 1,                # Sampling rounds
    "bfloat16": False,                     # Use bfloat16 (set True for TPU/A100)

    # ===== Logging =====
    "wandb": True,                         # Weights & Biases logging
    "project_name": "ChipPlacement_SDDS",  # W&B project name
}


# Example configurations for different scales
SMALL_CHIP_CONFIG = {
    **CHIP_PLACEMENT_CONFIG,
    "dataset_name": "Chip_10_components",
    "batch_size": 32,
    "N_basis_states": 20,
    "n_diffusion_steps": 30,
}

MEDIUM_CHIP_CONFIG = {
    **CHIP_PLACEMENT_CONFIG,
    "dataset_name": "Chip_20_components",
    "batch_size": 16,
    "N_basis_states": 10,
    "n_diffusion_steps": 50,
}

LARGE_CHIP_CONFIG = {
    **CHIP_PLACEMENT_CONFIG,
    "dataset_name": "Chip_50_components",
    "batch_size": 8,
    "N_basis_states": 5,
    "n_diffusion_steps": 100,
    "n_hidden_neurons": 128,
    "n_message_passes": 8,
}

# Dummy config for quick testing
DUMMY_CHIP_CONFIG = {
    **CHIP_PLACEMENT_CONFIG,
    "dataset_name": "Chip_dummy",
    "batch_size": 4,
    "N_basis_states": 5,
    "n_diffusion_steps": 10,
    "stop_epochs": 10,
    "n_message_passes": 2,
    "n_hidden_neurons": 32,
}


def get_chip_placement_config(scale="medium"):
    """
    Get chip placement configuration for specified scale.

    Args:
        scale: "small", "medium", "large", or "dummy"

    Returns:
        config dict
    """
    configs = {
        "small": SMALL_CHIP_CONFIG,
        "medium": MEDIUM_CHIP_CONFIG,
        "large": LARGE_CHIP_CONFIG,
        "dummy": DUMMY_CHIP_CONFIG,
    }

    if scale not in configs:
        raise ValueError(f"Unknown scale '{scale}'. Choose from {list(configs.keys())}")

    return configs[scale]


# Command line examples
COMMAND_LINE_EXAMPLES = """
# Small chip (10 components, fast training)
python argparse_ray_main.py \\
    --EnergyFunction ChipPlacement \\
    --dataset Chip_10_components \\
    --train_mode PPO \\
    --graph_mode normal \\
    --noise_potential gaussian \\
    --n_diffusion_steps 30 \\
    --batch_size 32 \\
    --n_basis_states 20 \\
    --GPUs 0

# Medium chip (20-50 components, recommended)
python argparse_ray_main.py \\
    --EnergyFunction ChipPlacement \\
    --dataset Chip_20_components \\
    --train_mode PPO \\
    --graph_mode normal \\
    --noise_potential gaussian \\
    --n_diffusion_steps 1000 \\
    --batch_size 32 \\
    --n_basis_states 20 \\
    --GPUs 0

# Dummy (quick test, 5 components)
python argparse_ray_main.py \\
    --EnergyFunction ChipPlacement \\
    --dataset Chip_dummy \\
    --train_mode PPO \\
    --graph_mode normal \\
    --noise_potential gaussian \\
    --n_diffusion_steps 10 \\
    --batch_size 4 \\
    --n_basis_states 5 \\
    --stop_epochs 10 \\
    --GPUs 0
"""

if __name__ == "__main__":
    print("=== Chip Placement SDDS Configuration ===\n")
    print("Default config (medium scale):")
    import pprint
    pprint.pprint(MEDIUM_CHIP_CONFIG)

    print("\n\n=== Command Line Examples ===")
    print(COMMAND_LINE_EXAMPLES)
