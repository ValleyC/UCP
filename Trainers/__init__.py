from .PPO_Trainer import PPO_Trainer

trainer_registry = {
    "PPO": PPO_Trainer,
}


def get_Trainer_class(config):
    """Get trainer class based on training mode."""
    train_mode = config["train_mode"]

    if train_mode not in trainer_registry:
        raise ValueError(f"Training mode '{train_mode}' not supported. Available: {list(trainer_registry.keys())}")

    return trainer_registry[train_mode]
