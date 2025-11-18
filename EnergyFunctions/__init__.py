from .ChipPlacementEnergy import ChipPlacementEnergyClass

energy_registry = {
    "ChipPlacement": ChipPlacementEnergyClass,
}


def get_Energy_class(config):
    """Get energy function class based on problem name."""
    problem_name = config["problem_name"]

    if problem_name not in energy_registry:
        raise ValueError(f"Problem '{problem_name}' is not supported. Available: {list(energy_registry.keys())}")

    return energy_registry[problem_name](config)