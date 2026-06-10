from dqn_agent import train_dqn, test_dqn
from plot_utility import plot_combined_training_curve, plot_test_distribution


# --- Run Configuration ---
RNG_SEED = 42
ENV_SEED = None  # if None, uses RNG_SEED for Gymnasium env.reset base
TRAIN_EPISODES = 10000
TEST_EPISODES = 100

MAP_CONFIGS = [
    {
        "size": 4,
        "name": "4x4",
        "desc": [
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG"
        ],
        "epsilon_decay_steps": 55_000,
        "epsilon_2_decay_steps": 35_000,
        "memory_size": 55_000,
    },
    {
        "size": 5,
        "name": "5x5",
        "desc": [
            "SFFFF",
            "FFFFF",
            "FHFHF",
            "FFFFH",
            "HFFFG"
        ],
        "epsilon_decay_steps": 90_000,
        "epsilon_2_decay_steps": 45_000,
        "memory_size": 90_000,
    },
    {
        "size": 6,
        "name": "6x6",
        "desc": [
            "SFFFFF",
            "FFFFFF",
            "FHFHFF",
            "FFFFFH",
            "FFFHFF",
            "HFFFFG"
        ],
        "epsilon_decay_steps": 110_000,
        "epsilon_2_decay_steps": 70_000,
        "memory_size": 110_000,
    },
    {
        "size": 7,
        "name": "7x7",
        "desc": [
            "SFFFFFF",
            "FFFFFFF",
            "FFFHFFF",
            "FFFFFHF",
            "FFFHFFF",
            "FFHFFHF",
            "HFFFFFG"
        ],
        "epsilon_decay_steps": 140_000,
        "epsilon_2_decay_steps": 85_000,
        "memory_size": 140_000,
    },
    {
        "size": 8,
        "name": "8x8",
        "desc": [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG"
        ],
        "epsilon_decay_steps": 180_000,
        "epsilon_2_decay_steps": 90_000,
        "memory_size": 180_000,
    },
]


def run_for_map(map_cfg, env_seed):
    map_name = map_cfg["name"]
    map_desc = map_cfg["desc"]
    epsilon_decay_steps = map_cfg["epsilon_decay_steps"]
    epsilon_2_decay_steps = map_cfg["epsilon_2_decay_steps"]
    memory_size = map_cfg["memory_size"]

    print(f"\n{'=' * 60}")
    print(f"FrozenLake {map_name}")
    print(f"{'=' * 60}")

    print("--- Training (DQN + classical planner) ---")
    agent_with_planner = train_dqn(
        episodes=TRAIN_EPISODES,
        use_planner=True,
        rng_seed=RNG_SEED,
        env_seed=env_seed,
        map_desc=map_desc,
        map_name=map_name,
        epsilon_decay_steps=epsilon_decay_steps,
        epsilon_2_decay_steps=epsilon_2_decay_steps,
        memory_size=memory_size,
    )

    print("--- Training (DQN without classical planner) ---")
    agent_without_planner = train_dqn(
        episodes=TRAIN_EPISODES,
        use_planner=False,
        rng_seed=RNG_SEED,
        env_seed=env_seed,
        map_desc=map_desc,
        map_name=map_name,
        epsilon_decay_steps=epsilon_decay_steps,
        epsilon_2_decay_steps=epsilon_2_decay_steps,
        memory_size=memory_size,
    )

    plot_combined_training_curve(
        agent_with_planner.training_rewards,
        agent_without_planner.training_rewards,
        title=f"FrozenLake {map_name} Training Curve - DQN with classical planner VS plain DQN",
        epsilon_with_planner=agent_with_planner.training_epsilon,
        epsilon_2_with_planner=agent_with_planner.training_epsilon_2,
        epsilon_without_planner=agent_without_planner.training_epsilon,
        stats_with_planner={
            "episodes": agent_with_planner.training_episodes,
            "training_time_s": agent_with_planner.training_time_seconds,
            "plan_time_s": agent_with_planner.plan_time_seconds,
            "plan_time_pct": agent_with_planner.plan_time_pct,
        },
        stats_without_planner={
            "episodes": agent_without_planner.training_episodes,
            "training_time_s": agent_without_planner.training_time_seconds,
        },
    )

    print(f"--- Evaluation (with planner) [{map_name}] ---")
    test_rewards = test_dqn(
        agent_with_planner,
        episodes=TEST_EPISODES,
        rng_seed=RNG_SEED,
        env_seed=env_seed,
        map_desc=map_desc,
        map_name=map_name,
    )
    plot_test_distribution(
        test_rewards,
        title=f"FrozenLake {map_name} Test Distribution - DQN + classical planner",
    )

    print(f"--- Evaluation (no planner) [{map_name}] ---")
    test_rewards = test_dqn(
        agent_without_planner,
        episodes=TEST_EPISODES,
        rng_seed=RNG_SEED,
        env_seed=env_seed,
        map_desc=map_desc,
        map_name=map_name,
    )
    plot_test_distribution(
        test_rewards,
        title=f"FrozenLake {map_name} Test Distribution - DQN with NO classical planner",
    )


def main():
    env_seed = ENV_SEED if ENV_SEED is not None else RNG_SEED

    for map_cfg in MAP_CONFIGS:
        run_for_map(map_cfg, env_seed)


if __name__ == "__main__":
    main()
