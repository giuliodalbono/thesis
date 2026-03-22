from dqn_agent import train_dqn, test_dqn


# --- Run Configuration ---
RNG_SEED = 42
ENV_SEED = None  # if None, uses RNG_SEED for Gymnasium env.reset base


def main():
    print("FrozenLake: DQN + classical planner")

    env_seed = ENV_SEED if ENV_SEED is not None else RNG_SEED

    print("--- Training ---")
    dqn_agent = train_dqn(episodes=100, use_planner=True, rng_seed=RNG_SEED, env_seed=env_seed)
    print("--- Evaluation ---")
    test_dqn(dqn_agent, episodes=100, rng_seed=RNG_SEED, env_seed=env_seed)

    print("FrozenLake: DQN with NO classical planner")

    print("--- Training ---")
    dqn_agent = train_dqn(episodes=150, use_planner=False, rng_seed=RNG_SEED, env_seed=env_seed)
    print("--- Evaluation ---")
    test_dqn(dqn_agent, episodes=100, rng_seed=RNG_SEED, env_seed=env_seed)


if __name__ == "__main__":
    main()
