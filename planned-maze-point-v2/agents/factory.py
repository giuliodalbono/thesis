"""Single place for converting an experiment preset into a fresh agent.

Each call creates independent weights, replay and optimizer state. Services
share this factory, not a live agent instance (except within a cumulative sweep).
"""
import numpy as np
from .dqn import DQNAgent, DQNHyperparameters, RecoveryConfig
from environments.common import ACTION_TABLE


def create_agent(config: object, state_size: int, action_size: int) -> DQNAgent:
    if config.agent_name != "dqn":
        raise ValueError("Unsupported agent: {}".format(config.agent_name))
    if action_size != len(ACTION_TABLE):
        raise ValueError("DQN action count does not match the PointMaze action table.")
    return DQNAgent(
        state_size,
        np.asarray(ACTION_TABLE),
        DQNHyperparameters(
            config.gamma, config.learning_rate, config.batch_size,
            config.replay_capacity, config.learning_starts_transitions,
            config.target_sync_updates,
        ),
        RecoveryConfig(
            config.stuck_detection_steps, config.opposite_recovery_steps,
            config.random_recovery_steps, config.minimum_directional_progress,
        ),
        config.seed,
    )
