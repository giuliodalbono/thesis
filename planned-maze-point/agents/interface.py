"""Episode-level contract: services never access an algorithm's tensor layout."""
from typing import Protocol
from .dqn import DQNAgent
from environments.common import flatten_observation


class ExperimentAgent(Protocol):
    stored_environment_steps: int
    updates: int
    recovery_events: int
    recovery_actions: int

    def reset_episode(self): ...
    def select_action(self, observation, training, epsilon=0.0): ...
    def planner_action(self, index, noise_std=0.0): ...
    def use_planner(self, probability): ...
    def observe(self, observation, action, reward, next_observation, terminated, truncated, info, config,
                is_planner_action=False): ...
    def diagnostics(self): ...


class DQNExperimentAgent(DQNAgent):
    """Adapter only: retain the existing DQN action/update/timeout semantics."""
    def select_action(self, observation, training, epsilon=0.0):
        return self.act(flatten_observation(observation), epsilon)

    def planner_action(self, index, noise_std=0.0):
        if noise_std:
            raise ValueError("DQN's discrete planner adapter does not support action noise")
        return index

    def use_planner(self, probability):
        return self.rng.random() < probability

    def observe(self, observation, action, reward, next_observation, terminated, truncated, info, config,
                is_planner_action=False):
        self.store((flatten_observation(observation), action, float(reward),
                    flatten_observation(next_observation), bool(terminated or truncated)))
        self.train_if_due(config.train_every_steps, config.updates_per_training_event)

    def diagnostics(self):
        return {}
