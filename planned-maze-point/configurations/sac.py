"""SAC/HER settings, independent of DQN's epsilon and recovery configuration."""
from dataclasses import dataclass
from pathlib import Path
from typing import Union
from .common import PointMazeConfigValidation


@dataclass(frozen=True)
class SACSettings:
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.001
    batch_size: int = 256
    replay_capacity: int = 200_000
    hidden_sizes: tuple = (256, 256)
    learning_starts_transitions: int = 1_000
    train_every_steps: int = 4
    updates_per_training_event: int = 1
    her_sampled_goals: int = 4
    # Fixed alpha is used by the stability preset. ``"auto"`` remains a
    # supported ablation, but it must be explicitly selected and recorded.
    entropy_coefficient: Union[str, float] = 0.1
    her_strategy: str = "future"
    device: str = "cpu"


@dataclass(frozen=True)
class PointMazeOpenSACConfig(SACSettings, PointMazeConfigValidation):
    env_id: str = "PointMaze_Open-v3"
    seed: int = 42
    max_episode_steps: int = 500
    warmup_episode_counts: tuple = (25, 50, 100, 200)
    evaluation_episodes: int = 500
    progress_every_episodes: int = 10
    torch_threads: int = 1
    output_dir: Path = Path(__file__).resolve().parents[2] / "resources" / "planned-maze-point"

    @property
    def agent_name(self):
        return "sac_her"

    def environment_slug(self):
        return "pointmaze_open"

    def validate(self):
        expected = "PointMaze_Medium-v3" if isinstance(self, PointMazeMediumSACConfig) else "PointMaze_Open-v3"
        if self.env_id != expected:
            raise ValueError("Use the SAC preset matching the environment")
        positive = (self.max_episode_steps, self.batch_size, self.replay_capacity,
                    self.train_every_steps, self.updates_per_training_event,
                    self.learning_starts_transitions, self.evaluation_episodes,
                    self.her_sampled_goals, self.torch_threads, self.progress_every_episodes)
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in positive):
            raise ValueError("SAC counts must be positive integers")
        if self.replay_capacity <= 2 * self.max_episode_steps:
            raise ValueError("HER replay must retain completed episodes while the next is collected")
        if not self.hidden_sizes or any(n < 1 for n in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive widths")
        if not (0 < self.gamma <= 1 and 0 < self.tau <= 1 and self.learning_rate > 0):
            raise ValueError("Invalid SAC learning parameters")
        if self.her_strategy != "future":
            raise ValueError("This protocol uses future HER")
        valid_auto_entropy = self.entropy_coefficient == "auto"
        valid_fixed_entropy = (
            isinstance(self.entropy_coefficient, (int, float))
            and not isinstance(self.entropy_coefficient, bool)
            and self.entropy_coefficient > 0
        )
        if not (valid_auto_entropy or valid_fixed_entropy):
            raise ValueError(
                "entropy_coefficient must be 'auto' or a positive fixed number"
            )
        if not self.warmup_episode_counts or any(
            not isinstance(n, int) or n < 1 for n in self.warmup_episode_counts
        ) or any(b <= a for a, b in zip(self.warmup_episode_counts, self.warmup_episode_counts[1:])):
            raise ValueError("Warm-up checkpoints must be positive, strictly increasing integers")
        if not isinstance(self.output_dir, Path):
            raise ValueError("output_dir must be a pathlib.Path")


@dataclass(frozen=True)
class PointMazeMediumSACConfig(PointMazeOpenSACConfig):
    env_id: str = "PointMaze_Medium-v3"
    max_episode_steps: int = 1_000
    learning_starts_transitions: int = 5_000
    warmup_episode_counts: tuple = (200, 300, 400, 600, 800, 1000)

    def environment_slug(self):
        return "pointmaze_medium"
