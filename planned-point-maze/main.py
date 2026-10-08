from __future__ import annotations

import numpy as np

from dqn_agent import ENV_ID, get_checkpoint_dir, test_dqn, train_dqn
from plot_utility import plot_combined_training_curve, plot_test_distribution, plot_training_curve
from sac_agent import test_sac, train_sac
from planner import DeterministicPointMazePlanner, PlannerConfig
from point_maze_wrappers import build_action_table


# --- Run Configuration ---
# Select which algorithm(s) to run.
RUN_DQN = False
RUN_SAC = True
RUN_BOTH = False
# Control train/test execution independently.
RUN_TRAIN = True
RUN_TEST = True

RNG_SEED = 42
ENV_SEED = None  # if None, uses RNG_SEED for Gymnasium env.reset base
TRAIN_EPISODES = 5_000
TEST_EPISODES = 5_000
ACTION_TABLE_MODE = "nine"  # five | nine | thirteen
MAX_EPISODE_STEPS = 600
ENABLE_PLANNER_WARM_UP = True
PLANNER_WARM_UP_EPISODES = 2_000
ENABLE_ALTERNATING_WARMUP = True

PLANNER_GRID_SIZE = 21
TRAIN_EVERY_STEPS = 1
# Number of post-warm-up evaluation episodes. Set equal to TEST_EPISODES by default.
WARMUP_TEST_EPISODES = TEST_EPISODES

# --- SAC Hyperparameters ---
SAC_TOTAL_TIMESTEPS = 2_000
SAC_LEARNING_RATE = 3e-4
SAC_BATCH_SIZE = 256
SAC_GAMMA = 0.99
SAC_TAU = 0.005
SAC_TRAIN_FREQ = 1
SAC_GRADIENT_STEPS = 1
SAC_ENT_COEF = "auto"
SAC_LEARNING_STARTS = 1_000
SAC_DEVICE = "auto"

MAP_LABEL = "Medium_sparse"
EPSILON_DECAY_FACTOR = 0.9998
MEMORY_SIZE = 200_000


def _smoke_validate_action_table() -> None:
    """Quick consistency check for discrete->continuous action mapping."""
    expected_counts = {
        "five": 4,
        "nine": 8,
        "thirteen": 12,
    }
    if ACTION_TABLE_MODE not in expected_counts:
        valid = ", ".join(sorted(expected_counts))
        raise ValueError(
            f"Unknown ACTION_TABLE_MODE={ACTION_TABLE_MODE!r}. Expected one of: {valid}."
        )

    table = build_action_table(ACTION_TABLE_MODE)
    expected = expected_counts[ACTION_TABLE_MODE]
    if len(table) != expected:
        raise ValueError(
            f"ACTION_TABLE_MODE={ACTION_TABLE_MODE!r} should create {expected} actions, got {len(table)}."
        )
    for idx, vec in enumerate(table):
        arr = np.asarray(vec, dtype=np.float32).reshape(2)
        if np.any(arr < -1.0) or np.any(arr > 1.0):
            raise ValueError(f"Action {idx} out of force bounds [-1, 1]: {arr}")


def _smoke_validate_planner() -> None:
    table = build_action_table(ACTION_TABLE_MODE)
    planner = DeterministicPointMazePlanner(
        action_table=table,
        config=PlannerConfig(grid_size=PLANNER_GRID_SIZE),
    )
    fake_obs = np.zeros(10, dtype=np.float32)
    fake_obs[:2] = np.array([0.0, 0.0], dtype=np.float32)
    fake_obs[-2:] = np.array([3.0, 3.0], dtype=np.float32)
    action = planner.first_action(fake_obs)
    if action is None or not 0 <= int(action) < len(table):
        raise ValueError(
            f"Planner smoke validation failed: invalid action {action!r} for table size {len(table)}."
        )


def run_main_experiment():
    run_dqn = RUN_DQN
    run_sac = RUN_SAC
    if RUN_BOTH:
        run_dqn = True
        run_sac = True

    if not run_dqn and not run_sac:
        raise ValueError("At least one algorithm must be enabled: RUN_DQN or RUN_SAC (or RUN_BOTH).")
    if not RUN_TRAIN and not RUN_TEST:
        raise ValueError("At least one stage must run: RUN_TRAIN or RUN_TEST.")

    env_seed = ENV_SEED if ENV_SEED is not None else RNG_SEED
    checkpoint_dir = get_checkpoint_dir()
    _smoke_validate_action_table()
    _smoke_validate_planner()

    print(f"\n{'=' * 60}")
    print("PointMaze Medium Sparse")
    print(f"{'=' * 60}")
    print(f"[config] env_id={ENV_ID} | action_mode={ACTION_TABLE_MODE} | map={MAP_LABEL}")
    print(
        "[config] "
        f"run_dqn={run_dqn} run_sac={run_sac} run_train={RUN_TRAIN} run_test={RUN_TEST} | "
        f"planner_warm_up={ENABLE_PLANNER_WARM_UP} | "
        f"warm_up_episodes={PLANNER_WARM_UP_EPISODES} | "
        f"alternating_warmup={ENABLE_ALTERNATING_WARMUP} | "
        f"planner_grid_size={PLANNER_GRID_SIZE} | "
        f"train_every_steps={TRAIN_EVERY_STEPS} | "
        f"warmup_test_episodes={WARMUP_TEST_EPISODES}"
    )
    x_offset_without_planner = PLANNER_WARM_UP_EPISODES if ENABLE_PLANNER_WARM_UP else 0

    planned_agent = None
    plain_agent = None
    planned_sac_model = None
    plain_sac_model = None
    planned_sac_metrics = None
    plain_sac_metrics = None

    if run_dqn and RUN_TRAIN:
        print("--- Training (planned DQN with planner warm-up) ---")
        planned_agent = train_dqn(
            episodes=TRAIN_EPISODES,
            rng_seed=RNG_SEED,
            env_seed=env_seed,
            env_id=ENV_ID,
            env_label=MAP_LABEL,
            action_table_mode=ACTION_TABLE_MODE,
            epsilon_decay_factor=EPSILON_DECAY_FACTOR,
            memory_size=MEMORY_SIZE,
            checkpoint_dir=checkpoint_dir,
            max_episode_steps=MAX_EPISODE_STEPS,
            use_planner_warm_up=ENABLE_PLANNER_WARM_UP,
            planner_warm_up_episodes=PLANNER_WARM_UP_EPISODES,
            enable_alternating_warmup=ENABLE_ALTERNATING_WARMUP,
            warmup_test_episodes=WARMUP_TEST_EPISODES,
            planner_grid_size=PLANNER_GRID_SIZE,
            train_every_steps=TRAIN_EVERY_STEPS,
        )

        print("--- Training (plain DQN baseline) ---")
        plain_agent = train_dqn(
            episodes=TRAIN_EPISODES,
            rng_seed=RNG_SEED,
            env_seed=env_seed,
            env_id=ENV_ID,
            env_label=MAP_LABEL,
            action_table_mode=ACTION_TABLE_MODE,
            epsilon_decay_factor=EPSILON_DECAY_FACTOR,
            memory_size=MEMORY_SIZE,
            checkpoint_dir=checkpoint_dir,
            max_episode_steps=MAX_EPISODE_STEPS,
            use_planner_warm_up=False,
            planner_warm_up_episodes=0,
            enable_alternating_warmup=False,
            warmup_test_episodes=0,
            planner_grid_size=PLANNER_GRID_SIZE,
            train_every_steps=TRAIN_EVERY_STEPS,
        )

        if planned_agent.planner_warmup_rewards and planned_agent.agent_eval_warmup_rewards:
            plot_combined_training_curve(
                planned_agent.planner_warmup_rewards,
                planned_agent.agent_eval_warmup_rewards,
                title=(
                    "PointMaze Medium Sparse Warm-up Train Curve - Planner VS No-Update Agent DQN "
                    f"(action mode: {ACTION_TABLE_MODE})"
                ),
                x_offset_without_planner=0,
                reward_label_with_planner="Reward (planner warm-up train episodes)",
                reward_label_without_planner="Reward (DQN warm-up eval episodes, no update)",
            )
        if planned_agent.warmup_test_rewards:
            plot_test_distribution(
                planned_agent.warmup_test_rewards,
                title=(
                    "PointMaze Medium Sparse Warm-up Test Distribution - Planned DQN "
                    f"(action mode: {ACTION_TABLE_MODE})"
                ),
            )
            plot_training_curve(
                planned_agent.warmup_test_rewards,
                title=(
                    "PointMaze Medium Sparse Warm-up Test Curve - Planned DQN "
                    f"(action mode: {ACTION_TABLE_MODE})"
                ),
            )

        plot_combined_training_curve(
            planned_agent.training_rewards,
            plain_agent.training_rewards,
            title=(
                "PointMaze Medium Sparse Train Curve - Planned DQN VS Plain DQN "
                f"(action mode: {ACTION_TABLE_MODE})"
            ),
            epsilon_with_planner=planned_agent.training_epsilon,
            epsilon_without_planner=plain_agent.training_epsilon,
            x_offset_without_planner=x_offset_without_planner,
            stats_with_planner={
                "episodes": planned_agent.training_episodes,
                "training_time_s": planned_agent.training_time_seconds,
            },
            stats_without_planner={
                "episodes": plain_agent.training_episodes,
                "training_time_s": plain_agent.training_time_seconds,
            },
        )
        plot_training_curve(
            planned_agent.training_rewards,
            title=f"PointMaze Medium Sparse Train Curve - Planned DQN (action mode: {ACTION_TABLE_MODE})",
            epsilon=planned_agent.training_epsilon,
            stats={
                "episodes": planned_agent.training_episodes,
                "training_time_s": planned_agent.training_time_seconds,
            },
        )
        plot_training_curve(
            plain_agent.training_rewards,
            title=f"PointMaze Medium Sparse Train Curve - Plain DQN (action mode: {ACTION_TABLE_MODE})",
            epsilon=plain_agent.training_epsilon,
            stats={
                "episodes": plain_agent.training_episodes,
                "training_time_s": plain_agent.training_time_seconds,
            },
        )

    if run_dqn and RUN_TEST:
        if planned_agent is None or plain_agent is None:
            print("[warn] Skipping DQN test because RUN_TRAIN is False and no trained DQN model is loaded.")
        else:
            print("--- Evaluation (planned DQN) ---")
            planned_test_rewards = test_dqn(
                planned_agent,
                episodes=TEST_EPISODES,
                rng_seed=RNG_SEED,
                env_seed=env_seed,
                env_id=ENV_ID,
                env_label=MAP_LABEL,
                action_table_mode=ACTION_TABLE_MODE,
                max_episode_steps=MAX_EPISODE_STEPS,
            )
            print("--- Evaluation (plain DQN) ---")
            plain_test_rewards = test_dqn(
                plain_agent,
                episodes=TEST_EPISODES,
                rng_seed=RNG_SEED,
                env_seed=env_seed,
                env_id=ENV_ID,
                env_label=MAP_LABEL,
                action_table_mode=ACTION_TABLE_MODE,
                max_episode_steps=MAX_EPISODE_STEPS,
            )

            plot_test_distribution(
                planned_test_rewards,
                title=f"PointMaze Medium Sparse Test Distribution - Planned DQN (action mode: {ACTION_TABLE_MODE})",
            )
            plot_test_distribution(
                plain_test_rewards,
                title=f"PointMaze Medium Sparse Test Distribution - Plain DQN (action mode: {ACTION_TABLE_MODE})",
            )
            plot_combined_training_curve(
                planned_test_rewards,
                plain_test_rewards,
                title=(
                    "PointMaze Medium Sparse Test Curve - Planned DQN VS Plain DQN "
                    f"(action mode: {ACTION_TABLE_MODE})"
                ),
                x_offset_without_planner=x_offset_without_planner,
            )
            plot_training_curve(
                planned_test_rewards,
                title=f"PointMaze Medium Sparse Test Curve - Planned DQN (action mode: {ACTION_TABLE_MODE})",
            )
            plot_training_curve(
                plain_test_rewards,
                title=f"PointMaze Medium Sparse Test Curve - Plain DQN (action mode: {ACTION_TABLE_MODE})",
            )

    if run_sac and RUN_TRAIN:
        print("--- Training (planned SAC with planner warm-up) ---")
        planned_sac_model, planned_sac_metrics = train_sac(
            total_timesteps=SAC_TOTAL_TIMESTEPS,
            rng_seed=RNG_SEED,
            env_seed=env_seed,
            env_id=ENV_ID,
            env_label=MAP_LABEL,
            action_table_mode=ACTION_TABLE_MODE,
            learning_rate=SAC_LEARNING_RATE,
            batch_size=SAC_BATCH_SIZE,
            gamma=SAC_GAMMA,
            tau=SAC_TAU,
            train_freq=SAC_TRAIN_FREQ,
            gradient_steps=SAC_GRADIENT_STEPS,
            ent_coef=SAC_ENT_COEF,
            learning_starts=SAC_LEARNING_STARTS,
            device=SAC_DEVICE,
            checkpoint_dir=checkpoint_dir,
            max_episode_steps=MAX_EPISODE_STEPS,
            use_planner_warm_up=ENABLE_PLANNER_WARM_UP,
            planner_warm_up_episodes=PLANNER_WARM_UP_EPISODES,
            planner_grid_size=PLANNER_GRID_SIZE,
            enable_alternating_warmup=ENABLE_ALTERNATING_WARMUP,
            warmup_test_episodes=WARMUP_TEST_EPISODES,
            training_variant="planned",
        )

        print("--- Training (plain SAC baseline) ---")
        plain_sac_model, plain_sac_metrics = train_sac(
            total_timesteps=SAC_TOTAL_TIMESTEPS,
            rng_seed=RNG_SEED,
            env_seed=env_seed,
            env_id=ENV_ID,
            env_label=MAP_LABEL,
            action_table_mode=ACTION_TABLE_MODE,
            learning_rate=SAC_LEARNING_RATE,
            batch_size=SAC_BATCH_SIZE,
            gamma=SAC_GAMMA,
            tau=SAC_TAU,
            train_freq=SAC_TRAIN_FREQ,
            gradient_steps=SAC_GRADIENT_STEPS,
            ent_coef=SAC_ENT_COEF,
            learning_starts=SAC_LEARNING_STARTS,
            device=SAC_DEVICE,
            checkpoint_dir=checkpoint_dir,
            max_episode_steps=MAX_EPISODE_STEPS,
            use_planner_warm_up=False,
            planner_warm_up_episodes=0,
            planner_grid_size=PLANNER_GRID_SIZE,
            enable_alternating_warmup=False,
            warmup_test_episodes=0,
            training_variant="plain",
        )

        if (
            planned_sac_metrics["planner_warmup_rewards"]
            and planned_sac_metrics["agent_eval_warmup_rewards"]
        ):
            plot_combined_training_curve(
                planned_sac_metrics["planner_warmup_rewards"],
                planned_sac_metrics["agent_eval_warmup_rewards"],
                title=(
                    "PointMaze Medium Sparse Warm-up Train Curve - Planner VS No-Update Agent SAC "
                    f"(action mode: {ACTION_TABLE_MODE})"
                ),
                x_offset_without_planner=0,
                reward_label_with_planner="Reward (planner warm-up train episodes)",
                reward_label_without_planner="Reward (SAC warm-up eval episodes, no update)",
            )

        if planned_sac_metrics["warmup_test_rewards"]:
            plot_test_distribution(
                planned_sac_metrics["warmup_test_rewards"],
                title=(
                    "PointMaze Medium Sparse Warm-up Test Distribution - Planned SAC "
                    f"(action mode: {ACTION_TABLE_MODE})"
                ),
            )
            plot_training_curve(
                planned_sac_metrics["warmup_test_rewards"],
                title=(
                    "PointMaze Medium Sparse Warm-up Test Curve - Planned SAC "
                    f"(action mode: {ACTION_TABLE_MODE})"
                ),
            )

        plot_combined_training_curve(
            planned_sac_metrics["training_rewards"],
            plain_sac_metrics["training_rewards"],
            title=(
                "PointMaze Medium Sparse Train Curve - Planned SAC VS Plain SAC "
                f"(action mode: {ACTION_TABLE_MODE})"
            ),
            x_offset_without_planner=x_offset_without_planner,
            stats_with_planner={
                "episodes": planned_sac_metrics["episodes"],
                "training_time_s": planned_sac_metrics["training_time_seconds"],
            },
            stats_without_planner={
                "episodes": plain_sac_metrics["episodes"],
                "training_time_s": plain_sac_metrics["training_time_seconds"],
            },
        )
        plot_training_curve(
            planned_sac_metrics["training_rewards"],
            title=f"PointMaze Medium Sparse Train Curve - Planned SAC (action mode: {ACTION_TABLE_MODE})",
            stats={
                "episodes": planned_sac_metrics["episodes"],
                "training_time_s": planned_sac_metrics["training_time_seconds"],
            },
        )
        plot_training_curve(
            plain_sac_metrics["training_rewards"],
            title=f"PointMaze Medium Sparse Train Curve - Plain SAC (action mode: {ACTION_TABLE_MODE})",
            stats={
                "episodes": plain_sac_metrics["episodes"],
                "training_time_s": plain_sac_metrics["training_time_seconds"],
            },
        )

    if run_sac and RUN_TEST:
        if planned_sac_model is None or plain_sac_model is None:
            print("[warn] Skipping SAC test because RUN_TRAIN is False and no trained SAC model is loaded.")
        else:
            print("--- Evaluation (planned SAC) ---")
            planned_sac_test_rewards = test_sac(
                planned_sac_model,
                episodes=TEST_EPISODES,
                rng_seed=RNG_SEED,
                env_seed=env_seed,
                env_id=ENV_ID,
                env_label=MAP_LABEL,
                max_episode_steps=MAX_EPISODE_STEPS,
                deterministic=True,
            )
            print("--- Evaluation (plain SAC) ---")
            plain_sac_test_rewards = test_sac(
                plain_sac_model,
                episodes=TEST_EPISODES,
                rng_seed=RNG_SEED,
                env_seed=env_seed,
                env_id=ENV_ID,
                env_label=MAP_LABEL,
                max_episode_steps=MAX_EPISODE_STEPS,
                deterministic=True,
            )
            plot_test_distribution(
                planned_sac_test_rewards,
                title=f"PointMaze Medium Sparse Test Distribution - Planned SAC (action mode: {ACTION_TABLE_MODE})",
            )
            plot_test_distribution(
                plain_sac_test_rewards,
                title=f"PointMaze Medium Sparse Test Distribution - Plain SAC (action mode: {ACTION_TABLE_MODE})",
            )
            plot_combined_training_curve(
                planned_sac_test_rewards,
                plain_sac_test_rewards,
                title=(
                    "PointMaze Medium Sparse Test Curve - Planned SAC VS Plain SAC "
                    f"(action mode: {ACTION_TABLE_MODE})"
                ),
                x_offset_without_planner=x_offset_without_planner,
            )
            plot_training_curve(
                planned_sac_test_rewards,
                title=f"PointMaze Medium Sparse Test Curve - Planned SAC (action mode: {ACTION_TABLE_MODE})",
            )
            plot_training_curve(
                plain_sac_test_rewards,
                title=f"PointMaze Medium Sparse Test Curve - Plain SAC (action mode: {ACTION_TABLE_MODE})",
            )


def main():
    run_main_experiment()


if __name__ == "__main__":
    main()

