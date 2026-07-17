from pathlib import Path

from dqn_agent import (
    get_checkpoint_dir,
    load_agent_checkpoint,
    test_dqn,
    train_dqn,
)
from plot_utility import plot_combined_training_curve, plot_test_distribution


CHECKPOINT_PATH = (
    "resources/planned-deterministic-frozen-lake/checkpoints/"
    "FrozenLake_10x10_planner_warmup.pt"
)
MODE = "test"  # "test" | "train_and_test"
RNG_SEED = 42
ENV_SEED = None
TEST_EPISODES = 5000
TRAIN_EPISODES = 6000
CHECKPOINT_DIR = get_checkpoint_dir()


def _resolve_checkpoint_path(path_str: str) -> Path:
    checkpoint_file = Path(path_str)
    if checkpoint_file.is_absolute():
        return checkpoint_file

    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / checkpoint_file,
        project_root / "resources" / "planned-deterministic-frozen-lake" / "checkpoints" / checkpoint_file.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _validate_mode(mode: str) -> None:
    if mode not in {"test", "train_and_test"}:
        raise ValueError("MODE must be 'test' or 'train_and_test'.")


def run_from_checkpoint():
    _validate_mode(MODE)
    checkpoint_file = _resolve_checkpoint_path(CHECKPOINT_PATH)
    agent, metadata = load_agent_checkpoint(checkpoint_file)

    map_name = metadata["map_name"]
    map_desc = metadata["map_desc"]
    if map_name is None or map_desc is None:
        raise ValueError("Checkpoint missing map metadata (map_name/map_desc).")

    env_seed = ENV_SEED if ENV_SEED is not None else RNG_SEED
    checkpoint_type = metadata.get("checkpoint_type")

    if MODE == "train_and_test":
        if checkpoint_type != "planner_warmup":
            raise ValueError(
                "MODE='train_and_test' requires a 'planner_warmup' checkpoint."
            )
        if not agent.use_planner:
            raise ValueError(
                "MODE='train_and_test' expects a planner-enabled agent checkpoint."
            )

        print(f"--- Resume training from checkpoint [{checkpoint_file}] ---")
        agent = train_dqn(
            episodes=TRAIN_EPISODES,
            use_planner=True,
            rng_seed=RNG_SEED,
            env_seed=env_seed,
            map_desc=map_desc,
            map_name=map_name,
            epsilon_decay_factor=agent.epsilon_decay_factor,
            epsilon_2_decay_factor=agent.epsilon_2_decay_factor,
            memory_size=agent.memory_size,
            planner_warm_up_episodes=0,
            checkpoint_dir=CHECKPOINT_DIR,
            initial_agent=agent,
        )

        plot_combined_training_curve(
            agent.training_rewards,
            [],
            title=f"FrozenLake {map_name} Resumed Train Curve - from warm-up checkpoint",
            epsilon_with_planner=agent.training_epsilon,
            epsilon_2_with_planner=agent.training_epsilon_2,
            stats_with_planner={
                "episodes": agent.training_episodes,
                "training_time_s": agent.training_time_seconds,
                "plan_time_s": agent.plan_time_seconds,
                "plan_time_pct": agent.plan_time_pct,
            },
            stats_without_planner=None,
        )

    print(f"--- Evaluation from checkpoint [{checkpoint_file}] ---")
    test_rewards = test_dqn(
        agent,
        episodes=TEST_EPISODES,
        rng_seed=RNG_SEED,
        env_seed=env_seed,
        map_desc=map_desc,
        map_name=map_name,
    )
    planner_suffix = "DQN + classical planner" if agent.use_planner else "plain DQN"
    plot_test_distribution(
        test_rewards,
        title=f"FrozenLake {map_name} Test Distribution - {planner_suffix} (from checkpoint)",
    )


if __name__ == "__main__":
    run_from_checkpoint()
