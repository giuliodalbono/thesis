"""Episode-based epsilon schedule shared by training and report plotting."""
import math
from configurations import PointMazeConfig


def epsilon_decay_horizon(config: PointMazeConfig, training_episodes: int) -> int:
    """Return the per-agent episode at which epsilon reaches its minimum."""
    if training_episodes == 0:
        return 0
    return max(1, int(math.ceil(training_episodes * config.epsilon_decay_fraction)))


def training_epsilon(
    config: PointMazeConfig,
    episode_index: int,
    training_episodes: int,
) -> float:
    """Linearly anneal epsilon over the configured fraction of this budget."""
    decay_horizon = epsilon_decay_horizon(config, training_episodes)
    # episode_index is zero-based: first normal-training episode uses epsilon_start,
    # and the final episode of the decay horizon uses epsilon_final.
    fraction = min(1.0, episode_index / max(1, decay_horizon - 1))
    return config.epsilon_start + fraction * (config.epsilon_final - config.epsilon_start)
