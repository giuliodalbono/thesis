from plot_utility import (
    plot_combined_training_curve,
    plot_single_training_curve,
    plot_test_distribution,
)
from ppo_agent import train_ppo, test_ppo
from sac_agent import train_sac, test_sac


# --- Run Configuration ---
RNG_SEED = 42
ENV_SEED = None  # if None, uses RNG_SEED for Gymnasium env.reset base
TRAIN_EPISODES = 10000  # Up to 6x6 3k, up to 15x15 6k, up to 17x17 10k
TEST_EPISODES = 5000

# Toggle experiments from here (comment/uncomment as needed).
RUN_DQN = False
RUN_PPO = False
RUN_SAC = True

# PPO configuration (planner-free by design).
PPO_EVAL_DETERMINISTIC = True
PPO_OUTPUT_SUBDIR = "ppo"

# SAC configuration (planner-free by design).
SAC_EVAL_DETERMINISTIC = True
SAC_OUTPUT_SUBDIR = "sac"

# Map names to run (must match MAP_CONFIGS "name" entries).
# SELECTED_MAPS = ["4x4", "5x5", "6x6", "7x7", "8x8"]
# Use ["ALL"] to run every map in MAP_CONFIGS.
SELECTED_MAPS = ["ALL"]

# Planner-only warm-up: fixed epsilons at 1.0, planned actions only, then normal training.
ENABLE_PLANNER_WARM_UP = True
PLANNER_WARM_UP_EPISODES = 3000  # up to 15x15 1500, up to 17x17 3k

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
        "epsilon_decay_factor": 0.999,
        "epsilon_2_decay_factor": 0.998,
        "memory_size": 100_000,
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
        "epsilon_decay_factor": 0.999,
        "epsilon_2_decay_factor": 0.998,
        "memory_size": 100_000,
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
        "epsilon_decay_factor": 0.999,
        "epsilon_2_decay_factor": 0.998,
        "memory_size": 100_000,
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
        "epsilon_decay_factor": 0.999,
        "epsilon_2_decay_factor": 0.998,
        "memory_size": 100_000,
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
        "epsilon_decay_factor": 0.999,
        "epsilon_2_decay_factor": 0.998,
        "memory_size": 100_000,
    },
    {
        "size": 9,
        "name": "9x9",
        "desc": [
            "SFFFFFFFF",
            "FFFFFFHFF",
            "FFFHFFFFF",
            "FFFFFHFFF",
            "FFFHFFHFF",
            "FHHFFFHFF",
            "FHFFHFHFF",
            "FFFHFFFFF",
            "FFFFFHFFG"
        ],
        "epsilon_decay_factor": 0.999,
        "epsilon_2_decay_factor": 0.998,
        "memory_size": 100_000,
    },
    {
        "size": 10,
        "name": "10x10",
        "desc": [
            "SFFFFFFFFF",
            "FFFFFFHFFF",
            "FFFHFFFFFF",
            "FFFFFHFFFF",
            "FFFHFFHFFF",
            "FHHFFFHFFF",
            "FHFFHFHFFF",
            "FFFHFFFFHF",
            "FFFFFHFFFF",
            "FFHFFFFHFG"
        ],
        "epsilon_decay_factor": 0.999,
        "epsilon_2_decay_factor": 0.998,
        "memory_size": 100_000,
    },
    {
        "size": 11,
        "name": "11x11",
        "desc": [
            "SFFFFHFFHFF",
            "FFFFFFHFFFF",
            "FFFHFFFFHFF",
            "FHFFFHFFFFF",
            "FFFHFFHFFFF",
            "FHHFFFHFFFF",
            "FHFFHFHFFFF",
            "FFFHFFFFHFF",
            "FFHFFHFFHFF",
            "FFHFFFFHFFF",
            "FFFFHFFFHFG"
        ],
        "epsilon_decay_factor": 0.999,
        "epsilon_2_decay_factor": 0.998,
        "memory_size": 100_000,
    },
    {
        "size": 12,
        "name": "12x12",
        "desc": [
            "SFFFFHFFHFFF",
            "FFFFFFHFFFFF",
            "FFFHFFFFHFFF",
            "FHFFFHFFFFFF",
            "FFFHFFHFFFFF",
            "FHHFFFHFFFFF",
            "FHFFHFHFFHFF",
            "FFFHFFFFHFFF",
            "FFHFFHFFHFFF",
            "FFHFFFFHFFFF",
            "FFFFHFFFHFFF",
            "FFHFFFHFFHFG"
        ],
        "epsilon_decay_factor": 0.999,
        "epsilon_2_decay_factor": 0.998,
        "memory_size": 100_000,
    },
    {
        "size": 13,
        "name": "13x13",
        "desc": [
            "SFFFFHFFHFFHF",
            "FFFHFFHFFFFFF",
            "FFFHFFFFHFFFF",
            "FHFFFHFFFHFFH",
            "FFFHFFHFFFFFF",
            "FHHFFFHFFHFFF",
            "FHFFHFHFFHFFF",
            "FFFHFFFFHFFFF",
            "FFHFFHFFHFFHF",
            "FFHFFFFHFFFHF",
            "FFFFHFFFHFFFF",
            "FFHFFFHFFHFFF",
            "FFFHFHFFFHFFG"
        ],
        "epsilon_decay_factor": 0.999,
        "epsilon_2_decay_factor": 0.998,
        "memory_size": 100_000,
    },
    {
        "size": 14,
        "name": "14x14",
        "desc": [
            "SFFFFHFFHFFHFF",
            "FFFHFFHFFFFFFF",
            "FFFHFFFFHFFFFF",
            "FHFFFHFFFHFFHF",
            "FFFHFFHFFFFFFF",
            "FHHFFFHFFHFFFF",
            "FHFFHFHFFHFFFF",
            "FFFHFFFFHFFFFF",
            "FFHFFHFFHFFHFF",
            "FFHFFFFHFFFHFF",
            "FFFFHFFFHFFFFF",
            "FFHFFFHFFHFFFF",
            "FFFHFHFFFHFFHF",
            "FFFFHFFFHFFFFG"
        ],
        "epsilon_decay_factor": 0.999,
        "epsilon_2_decay_factor": 0.998,
        "memory_size": 100_000,
    },
    {
        "size": 15,
        "name": "15x15",
        "desc": [
            "SFFFFHFFHFFHFFF",
            "FFFHFFHFFFFFFFF",
            "FFFHFFFFHFFFFFF",
            "FHFFFHFFFHFFHFF",
            "FFFHFFHFFFFFFFF",
            "FHHFFFHFFHFFFFF",
            "FHFFHFHFFHFFFFF",
            "FFFHFFFFHFFFFFF",
            "FFHFFHFFHFFHFFF",
            "FFHFFFFHFFFHFFF",
            "FFFFHFFFHFFFFFF",
            "FFHFFFHFFHFFFFF",
            "FFFHFHFFFHFFHFF",
            "FFFFHFFFHFHFFFH",
            "FFFFFFHFFFFHFFG"
        ],
        "epsilon_decay_factor": 0.999,
        "epsilon_2_decay_factor": 0.998,
        "memory_size": 100_000,
    },
    {
        "size": 16,
        "name": "16x16",
        "desc": [
            "SFFFFHFFHFFHFFFF",
            "FFFHFFHFFFFFFFFF",
            "FFFHFFFFHFFFFFFF",
            "FHFFFHFFFHFFHFFF",
            "FFFHFFHFFFFFFFFF",
            "FHHFFFHFFHFFFFFF",
            "FHFFHFHFFHFFFFFF",
            "FFFHFFFFHFFFFFFF",
            "FFHFFHFFHFFHFFFF",
            "FFHFFFFHFFFHFFFF",
            "FFFFHFFFHFFFFFFF",
            "FFHFFFHFFHFFFFFF",
            "FFFHFHFFFHFFHFFF",
            "FFFFHFFFHFHFFFHF",
            "FFFFFFHFFFFHFFFF",
            "FFFFFFFFFHFFHFFG"
        ],
        "epsilon_decay_factor": 0.999,
        "epsilon_2_decay_factor": 0.998,
        "memory_size": 100_000,
    },
    {
        "size": 17,
        "name": "17x17",
        "desc": [
            "SFFFFHFFHFFHFFFFF",
            "FFFHFFHFFFFFFFFFF",
            "FFFHFFFFHFFFFFFFF",
            "FHFFFHFFFHFFHFFFF",
            "FFFHFFHFFFFFFFFFF",
            "FHHFFFHFFHFFFFFFF",
            "FHFFHFHFFHFFFFFFF",
            "FFFHFFFFHFFFFFFFF",
            "FFHFFHFFHFFHFFFFF",
            "FFHFFFFHFFFHFFFFF",
            "FFFFHFFFHFFFFFFFF",
            "FFHFFFHFFHFFFFHFF",
            "FFFHFHFFFHFFHFFFF",
            "FFFFHFFFHFHFFFHFF",
            "FFFFFFHFFFFHFFFFF",
            "FFFFFFFFFHFFHFFFF",
            "FFFFFHFFFHFFFFFFG"
        ],
        "epsilon_decay_factor": 0.999,
        "epsilon_2_decay_factor": 0.998,
        "memory_size": 100_000,
    }
]

PPO_PRESETS = {
    "small": {
        "size_range": (4, 8),
        "label": "small",
        "total_timesteps": 150_000,
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 64,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
    },
    "medium": {
        "size_range": (9, 12),
        "label": "medium",
        "total_timesteps": 400_000,
        "learning_rate": 2.5e-4,
        "n_steps": 2048,
        "batch_size": 128,
        "gamma": 0.995,
        "ent_coef": 0.02,
        "gae_lambda": 0.97,
        "clip_range": 0.2,
    },
    "large": {
        "size_range": (13, 17),
        "label": "large",
        "total_timesteps": 1_000_000,
        "learning_rate": 1e-4,
        "n_steps": 4096,
        "batch_size": 256,
        "gamma": 0.995,
        "ent_coef": 0.03,
        "gae_lambda": 0.98,
        "clip_range": 0.15,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
}


def _ppo_preset_for_map_size(size):
    for preset in PPO_PRESETS.values():
        lo, hi = preset["size_range"]
        if lo <= size <= hi:
            return preset
    raise ValueError(f"No PPO preset configured for map size {size}.")


SAC_PRESETS = {
    "small": {
        "size_range": (4, 8),
        "label": "small",
        "total_timesteps": 200_000,
        "learning_rate": 3e-4,
        "batch_size": 128,
        "gamma": 0.99,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
    },
    "medium": {
        "size_range": (9, 12),
        "label": "medium",
        "total_timesteps": 500_000,
        "learning_rate": 2e-4,
        "batch_size": 256,
        "gamma": 0.995,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
    },
    "large": {
        "size_range": (13, 17),
        "label": "large",
        "total_timesteps": 1_200_000,
        "learning_rate": 1e-4,
        "batch_size": 256,
        "gamma": 0.995,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto_0.1",
    },
}


def _sac_preset_for_map_size(size):
    for preset in SAC_PRESETS.values():
        lo, hi = preset["size_range"]
        if lo <= size <= hi:
            return preset
    raise ValueError(f"No SAC preset configured for map size {size}.")


def run_for_map(map_cfg, env_seed, enable_planner_warm_up):
    from dqn_agent import get_checkpoint_dir, load_agent_checkpoint, train_dqn, test_dqn

    map_name = map_cfg["name"]
    map_desc = map_cfg["desc"]
    epsilon_decay_factor = map_cfg["epsilon_decay_factor"]
    epsilon_2_decay_factor = map_cfg["epsilon_2_decay_factor"]
    memory_size = map_cfg["memory_size"]
    checkpoint_dir = get_checkpoint_dir()

    print(f"\n{'=' * 60}")
    print(f"FrozenLake {map_name}")
    print(f"{'=' * 60}")

    def _evaluate_after_warmup(**kwargs):
        checkpoint_file = kwargs.get("checkpoint_path")
        if checkpoint_file is None:
            print("[warm-up test] checkpoint unavailable, skipping warm-up evaluation.")
            return

        warmup_agent, _ = load_agent_checkpoint(checkpoint_file)
        print(f"--- Evaluation right after planner warm-up [{map_name}] ---")
        warmup_test_rewards = test_dqn(
            warmup_agent,
            episodes=TEST_EPISODES,
            rng_seed=RNG_SEED,
            env_seed=env_seed,
            map_desc=map_desc,
            map_name=f"{map_name} warm-up",
        )
        plot_test_distribution(
            warmup_test_rewards,
            title=(
                f"FrozenLake {map_name} Warm-up Test Distribution - "
                f"DQN + classical planner"
            ),
        )
        plot_combined_training_curve(
            warmup_test_rewards,
            [],
            title=f"FrozenLake {map_name} Warm-up Test Curve - DQN + planner",
            stats_with_planner=None,
            stats_without_planner=None,
        )

    print("--- Training (DQN + classical planner) ---")
    agent_with_planner = train_dqn(
        episodes=TRAIN_EPISODES,
        use_planner=True,
        rng_seed=RNG_SEED,
        env_seed=env_seed,
        map_desc=map_desc,
        map_name=map_name,
        epsilon_decay_factor=epsilon_decay_factor,
        epsilon_2_decay_factor=epsilon_2_decay_factor,
        memory_size=memory_size,
        planner_warm_up_episodes=PLANNER_WARM_UP_EPISODES if enable_planner_warm_up else 0,
        checkpoint_dir=checkpoint_dir,
        on_planner_warmup_complete=_evaluate_after_warmup if enable_planner_warm_up else None,
    )

    print("--- Training (DQN without classical planner) ---")
    agent_without_planner = train_dqn(
        episodes=TRAIN_EPISODES,
        use_planner=False,
        rng_seed=RNG_SEED,
        env_seed=env_seed,
        map_desc=map_desc,
        map_name=map_name,
        epsilon_decay_factor=epsilon_decay_factor,
        epsilon_2_decay_factor=epsilon_2_decay_factor,
        memory_size=memory_size,
        checkpoint_dir=checkpoint_dir,
    )

    warm_up_suffix = (
        f" (planner warm-up {PLANNER_WARM_UP_EPISODES} ep)"
        if enable_planner_warm_up
        else ""
    )
    x_offset_without_planner = PLANNER_WARM_UP_EPISODES if enable_planner_warm_up else 0

    plot_combined_training_curve(
        agent_with_planner.training_rewards,
        agent_without_planner.training_rewards,
        title=(
            f"FrozenLake {map_name} Train Curve - "
            f"DQN + planner VS plain DQN{warm_up_suffix}"
        ),
        epsilon_with_planner=agent_with_planner.training_epsilon,
        epsilon_2_with_planner=agent_with_planner.training_epsilon_2,
        epsilon_without_planner=agent_without_planner.training_epsilon,
        x_offset_without_planner=x_offset_without_planner,
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
    test_planner_rewards = test_dqn(
        agent_with_planner,
        episodes=TEST_EPISODES,
        rng_seed=RNG_SEED,
        env_seed=env_seed,
        map_desc=map_desc,
        map_name=map_name,
    )
    plot_test_distribution(
        test_planner_rewards,
        title=f"FrozenLake {map_name} Test Distribution - DQN + classical planner",
    )

    print(f"--- Evaluation (no planner) [{map_name}] ---")
    test_no_planner_rewards = test_dqn(
        agent_without_planner,
        episodes=TEST_EPISODES,
        rng_seed=RNG_SEED,
        env_seed=env_seed,
        map_desc=map_desc,
        map_name=map_name,
    )
    plot_test_distribution(
        test_no_planner_rewards,
        title=f"FrozenLake {map_name} Test Distribution - plain DQN",
    )
    plot_combined_training_curve(
        test_planner_rewards,
        test_no_planner_rewards,
        title=(
            f"FrozenLake {map_name} Test Curve - "
            f"DQN + planner VS plain DQN"
        ),
        stats_with_planner=None,
        stats_without_planner=None
    )


def run_ppo_for_map(map_cfg, env_seed):
    map_size = map_cfg["size"]
    map_name = map_cfg["name"]
    map_desc = map_cfg["desc"]
    preset = _ppo_preset_for_map_size(map_size)

    print(f"\n{'=' * 60}")
    print(f"FrozenLake {map_name} (PPO)")
    print(f"{'=' * 60}")

    print("--- Training (PPO) ---")
    ppo_model, ppo_train_metrics = train_ppo(
        total_timesteps=preset["total_timesteps"],
        rng_seed=RNG_SEED,
        env_seed=env_seed,
        map_desc=map_desc,
        map_name=map_name,
        learning_rate=preset["learning_rate"],
        n_steps=preset["n_steps"],
        batch_size=preset["batch_size"],
        gamma=preset["gamma"],
        ent_coef=preset["ent_coef"],
        gae_lambda=preset["gae_lambda"],
        clip_range=preset["clip_range"],
        vf_coef=preset.get("vf_coef", 0.5),
        max_grad_norm=preset.get("max_grad_norm", 0.5),
    )

    train_rewards = ppo_train_metrics["training_rewards"]
    plot_single_training_curve(
        train_rewards,
        title=f"FrozenLake {map_name} Train Curve - PPO",
        stats={
            "episodes": ppo_train_metrics["episodes"],
            "training_time_s": ppo_train_metrics["training_time_seconds"],
        },
        output_subdir=PPO_OUTPUT_SUBDIR,
    )

    print(f"--- Evaluation (PPO) [{map_name}] ---")
    test_rewards = test_ppo(
        ppo_model,
        episodes=TEST_EPISODES,
        rng_seed=RNG_SEED,
        env_seed=env_seed,
        map_desc=map_desc,
        map_name=map_name,
        deterministic=PPO_EVAL_DETERMINISTIC,
    )
    plot_test_distribution(
        test_rewards,
        title=f"FrozenLake {map_name} Test Distribution - PPO",
        output_subdir=PPO_OUTPUT_SUBDIR,
    )
    plot_single_training_curve(
        test_rewards,
        title=f"FrozenLake {map_name} Test Curve - PPO",
        output_subdir=PPO_OUTPUT_SUBDIR,
    )


def run_sac_for_map(map_cfg, env_seed):
    map_size = map_cfg["size"]
    map_name = map_cfg["name"]
    map_desc = map_cfg["desc"]
    preset = _sac_preset_for_map_size(map_size)

    print(f"\n{'=' * 60}")
    print(f"FrozenLake {map_name} (SAC)")
    print(f"{'=' * 60}")

    print("--- Training (SAC) ---")
    sac_model, sac_train_metrics = train_sac(
        total_timesteps=preset["total_timesteps"],
        rng_seed=RNG_SEED,
        env_seed=env_seed,
        map_desc=map_desc,
        map_name=map_name,
        learning_rate=preset["learning_rate"],
        batch_size=preset["batch_size"],
        gamma=preset["gamma"],
        tau=preset["tau"],
        train_freq=preset["train_freq"],
        gradient_steps=preset["gradient_steps"],
        ent_coef=preset["ent_coef"],
    )

    train_rewards = sac_train_metrics["training_rewards"]
    plot_single_training_curve(
        train_rewards,
        title=f"FrozenLake {map_name} Train Curve - SAC",
        stats={
            "episodes": sac_train_metrics["episodes"],
            "training_time_s": sac_train_metrics["training_time_seconds"],
        },
        output_subdir=SAC_OUTPUT_SUBDIR,
    )

    print(f"--- Evaluation (SAC) [{map_name}] ---")
    test_rewards = test_sac(
        sac_model,
        episodes=TEST_EPISODES,
        rng_seed=RNG_SEED,
        env_seed=env_seed,
        map_desc=map_desc,
        map_name=map_name,
        deterministic=SAC_EVAL_DETERMINISTIC,
    )
    plot_test_distribution(
        test_rewards,
        title=f"FrozenLake {map_name} Test Distribution - SAC",
        output_subdir=SAC_OUTPUT_SUBDIR,
    )
    plot_single_training_curve(
        test_rewards,
        title=f"FrozenLake {map_name} Test Curve - SAC",
        output_subdir=SAC_OUTPUT_SUBDIR,
    )


def _selected_map_configs():
    if len(SELECTED_MAPS) == 1 and SELECTED_MAPS[0].upper() == "ALL":
        return MAP_CONFIGS

    configs_by_name = {cfg["name"]: cfg for cfg in MAP_CONFIGS}
    missing = [name for name in SELECTED_MAPS if name not in configs_by_name]
    if missing:
        available = ", ".join(configs_by_name)
        raise ValueError(f"Unknown map name(s) in SELECTED_MAPS: {missing}. Available: {available}")
    return [configs_by_name[name] for name in SELECTED_MAPS]


def main():
    env_seed = ENV_SEED if ENV_SEED is not None else RNG_SEED

    for map_cfg in _selected_map_configs():
        if RUN_DQN:
            run_for_map(map_cfg, env_seed, enable_planner_warm_up=ENABLE_PLANNER_WARM_UP)
        if RUN_PPO:
            run_ppo_for_map(map_cfg, env_seed)
        if RUN_SAC:
            run_sac_for_map(map_cfg, env_seed)


if __name__ == "__main__":
    main()
