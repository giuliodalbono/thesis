from pathlib import Path


class PointMazeConfigValidation:
    """Shared validation for the two environment-specific PointMaze configs."""

    def _validate_common(self) -> None:
        if not self.warmup_episode_counts or min(self.warmup_episode_counts) < 1:
            raise ValueError("warmup_episode_counts must contain positive episode counts")
        if any(
            following <= previous
            for previous, following in zip(
                self.warmup_episode_counts, self.warmup_episode_counts[1:]
            )
        ):
            raise ValueError("warmup_episode_counts must be strictly increasing")
        if self.evaluation_episodes < 1:
            raise ValueError("evaluation_episodes must be positive")
        if self.train_every_steps < 1 or self.updates_per_training_event < 1:
            raise ValueError("DQN training frequencies must be positive")
        if not 0.0 < self.epsilon_decay_fraction <= 1.0:
            raise ValueError("epsilon_decay_fraction must be in (0, 1]")
        if not 0.0 <= self.epsilon_final <= self.epsilon_start:
            raise ValueError("epsilon values must satisfy 0 <= final <= start")
        if min(self.stuck_detection_steps, self.opposite_recovery_steps, self.random_recovery_steps) < 1:
            raise ValueError("Recovery step counts must be positive")
        if not isinstance(self.output_dir, Path):
            raise ValueError("output_dir must be a pathlib.Path")

    @property
    def environment(self):
        """The preset binds agent hyperparameters to one environment and planner."""
        from environments import environment_for
        return environment_for(self.env_id)

    @property
    def agent_name(self):
        return "dqn"

    @property
    def planner_name(self):
        return self.environment.planner_name

    def to_dict(self):
        """JSON-safe complete preset, including the selected implementations."""
        from dataclasses import asdict
        values = {key: str(value) if isinstance(value, Path) else value
                  for key, value in asdict(self).items()}
        values.update(agent_name=self.agent_name, planner_name=self.planner_name,
                      configuration_class=type(self).__name__)
        return values
