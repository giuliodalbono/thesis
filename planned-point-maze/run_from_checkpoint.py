from __future__ import annotations

from pathlib import Path

from dqn_agent import (
    ENV_ID,
    get_checkpoint_dir,
    load_agent_checkpoint,
    test_dqn,
    train_dqn,
)
from plot_utility import plot_test_distribution, plot_training_curve
from sac_agent import load_sac_checkpoint, test_sac, train_sac


AGENT_KIND = "dqn"  # "dqn" | "sac"
CHECKPOINT_PATH = "resources/planned-point-maze/checkpoints/PointMaze_Medium_sparse_nine_dqn_planner_warmup.pt"
MODE = "test"  # "test" | "train_and_test"
RNG_SEED = 42
ENV_SEED = None
TEST_EPISODES = 1_000
TRAIN_EPISODES = 5_000
CHECKPOINT_DIR = get_checkpoint_dir()
MAX_EPISODE_STEPS = None
SAC_TOTAL_TIMESTEPS = 25_000


def _resolve_checkpoint_path(path_str: str) -> Path:
    checkpoint_file = Path(path_str)
    if checkpoint_file.is_absolute():
        return checkpoint_file

    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / checkpoint_file,
        project_root / "resources" / "planned-point-maze" / "checkpoints" / checkpoint_file.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _validate_mode(mode: str) -> None:
    if mode not in {"test", "train_and_test"}:
        raise ValueError("MODE must be 'test' or 'train_and_test'.")


def _validate_agent_kind(kind: str) -> None:
    if kind not in {"dqn", "sac"}:
        raise ValueError("AGENT_KIND must be 'dqn' or 'sac'.")


def run_from_checkpoint():
    _validate_mode(MODE)
    _validate_agent_kind(AGENT_KIND)
    checkpoint_file = _resolve_checkpoint_path(CHECKPOINT_PATH)
    env_seed = ENV_SEED if ENV_SEED is not None else RNG_SEED
    if AGENT_KIND == "dqn":
        agent, metadata = load_agent_checkpoint(checkpoint_file)
        env_id = metadata.get("env_id", ENV_ID)
        env_label = metadata.get("env_label", "Medium_sparse")
        action_table_mode = metadata.get("action_table_mode", "nine")
        planner_warm_up_episodes = int(metadata.get("planner_warm_up_episodes", 0))
        use_planner_warm_up = bool(metadata.get("use_planner_warm_up", False))
        planner_grid_size = int(metadata.get("planner_grid_size", 21))

        if MODE == "train_and_test":
            print(f"--- Resume training (DQN) from checkpoint [{checkpoint_file}] ---")
            agent = train_dqn(
                episodes=TRAIN_EPISODES,
                rng_seed=RNG_SEED,
                env_seed=env_seed,
                env_id=env_id,
                env_label=env_label,
                action_table_mode=action_table_mode,
                epsilon_decay_factor=agent.epsilon_decay_factor,
                memory_size=agent.memory_size,
                checkpoint_dir=CHECKPOINT_DIR,
                max_episode_steps=MAX_EPISODE_STEPS,
                use_planner_warm_up=use_planner_warm_up,
                planner_warm_up_episodes=planner_warm_up_episodes,
                planner_grid_size=planner_grid_size,
            )
            plot_training_curve(
                agent.training_rewards,
                title=f"PointMaze {env_label} Resumed Train Curve - DQN",
                epsilon=agent.training_epsilon,
                stats={
                    "episodes": agent.training_episodes,
                    "training_time_s": agent.training_time_seconds,
                },
            )

        print(f"--- Evaluation (DQN) from checkpoint [{checkpoint_file}] ---")
        test_rewards = test_dqn(
            agent,
            episodes=TEST_EPISODES,
            rng_seed=RNG_SEED,
            env_seed=env_seed,
            env_id=env_id,
            env_label=env_label,
            action_table_mode=action_table_mode,
            max_episode_steps=MAX_EPISODE_STEPS,
        )
        plot_test_distribution(
            test_rewards,
            title=f"PointMaze {env_label} Test Distribution - DQN (from checkpoint)",
        )
        return

    model, metadata = load_sac_checkpoint(str(checkpoint_file))
    env_id = metadata.get("env_id", ENV_ID)
    env_label = metadata.get("env_label", "Medium_sparse")
    action_table_mode = metadata.get("action_table_mode", "nine")
    variant = metadata.get("training_variant", "planned")
    if MODE == "train_and_test":
        print(
            f"--- Resume training (SAC) from checkpoint [{checkpoint_file}] --- "
            "(resume starts from model weights; replay buffer is rebuilt)."
        )
        model, metrics = train_sac(
            total_timesteps=SAC_TOTAL_TIMESTEPS,
            rng_seed=RNG_SEED,
            env_seed=env_seed,
            env_id=env_id,
            env_label=env_label,
            action_table_mode=action_table_mode,
            checkpoint_dir=CHECKPOINT_DIR,
            max_episode_steps=MAX_EPISODE_STEPS,
            use_planner_warm_up=False,
            planner_warm_up_episodes=0,
            enable_alternating_warmup=False,
            warmup_test_episodes=0,
            training_variant=f"{variant}_resumed",
        )
        plot_training_curve(
            metrics["training_rewards"],
            title=f"PointMaze {env_label} Resumed Train Curve - SAC",
            stats={
                "episodes": metrics["episodes"],
                "training_time_s": metrics["training_time_seconds"],
            },
        )

    print(f"--- Evaluation (SAC) from checkpoint [{checkpoint_file}] ---")
    test_rewards = test_sac(
        model,
        episodes=TEST_EPISODES,
        rng_seed=RNG_SEED,
        env_seed=env_seed,
        env_id=env_id,
        env_label=env_label,
        max_episode_steps=MAX_EPISODE_STEPS,
        deterministic=True,
    )
    plot_test_distribution(
        test_rewards,
        title=f"PointMaze {env_label} Test Distribution - SAC (from checkpoint)",
    )


if __name__ == "__main__":
    run_from_checkpoint()

