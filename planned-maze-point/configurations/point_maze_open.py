from dataclasses import dataclass
from pathlib import Path

from .common import PointMazeConfigValidation


@dataclass(frozen=True)
class PointMazeOpenConfig(PointMazeConfigValidation):
    """Existing Open DQN preset, preserved during the architecture refactor."""

    env_id: str = "PointMaze_Open-v3"
    seed: int = 42
    max_episode_steps: int = 500
    warmup_episode_counts: tuple = (25, 50, 100, 200)
    evaluation_episodes: int = 5_000

    replay_capacity: int = 100_000
    batch_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    target_sync_updates: int = 2_000
    learning_starts_transitions: int = 1_000
    train_every_steps: int = 4
    updates_per_training_event: int = 1
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_fraction: float = 0.75

    stuck_detection_steps: int = 10
    opposite_recovery_steps: int = 15
    random_recovery_steps: int = 15
    minimum_directional_progress: float = 0.002

    output_dir: Path = Path(__file__).resolve().parents[2] / "resources" / "planned-maze-point"

    def validate(self) -> None:
        if self.env_id != "PointMaze_Open-v3":
            raise ValueError("Use the configuration class matching the environment")
        self._validate_common()

    def environment_slug(self) -> str:
        return "pointmaze_open"
