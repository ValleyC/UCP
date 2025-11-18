from .GaussianNoise import GaussianNoiseDistr

noise_distribution_registry = {
    "gaussian": GaussianNoiseDistr,
}


def get_Noise_class(config):
    """Get noise distribution class for continuous diffusion."""
    noise_distr_str = config["noise_potential"]

    if noise_distr_str not in noise_distribution_registry:
        raise ValueError(f"Noise distribution '{noise_distr_str}' not supported. Available: {list(noise_distribution_registry.keys())}")

    return noise_distribution_registry[noise_distr_str](config)
