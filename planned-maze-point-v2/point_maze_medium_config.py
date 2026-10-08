from dataclasses import dataclass
from pathlib import Path

from point_maze_config_common import PointMazeConfigValidation


@dataclass(frozen=True)
class PointMazeMediumConfig(PointMazeConfigValidation):
    """Larger initial configuration for the obstacle-constrained Medium maze.

    The training and warm-up budgets are intentionally larger than Open because
    demonstrations and exploration must cover obstacle-constrained routes.
    """

    env_id: str = "PointMaze_Medium-v3"
    seed: int = 42
    max_episode_steps: int = 1_000
    warmup_episode_counts: tuple = (25_000, 30_000, 35_000, 40_000, 45_000, 50_000)
    evaluation_episodes: int = 5_000

    replay_capacity: int = 200_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    target_sync_updates: int = 5_000
    learning_starts_transitions: int = 5_000
    train_every_steps: int = 4
    updates_per_training_event: int = 1
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_fraction: float = 0.75

    stuck_detection_steps: int = 10
    opposite_recovery_steps: int = 15
    random_recovery_steps: int = 15
    minimum_directional_progress: float = 0.002

    output_dir: Path = Path(__file__).resolve().parents[1] / "resources" / "planned-maze-point-v2"

    def validate(self) -> None:
        self._validate_common()

    def environment_slug(self) -> str:
        return "pointmaze_medium"
