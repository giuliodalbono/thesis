from dqn_agent import train_dqn, test_dqn
from plot_utility import plot_test_distribution, plot_training_curve


# --- Run Configuration ---
RNG_SEED = 42
ENV_SEED = None  # if None, uses RNG_SEED for Gymnasium env.reset base
TRAIN_EPISODES = 10000
TEST_EPISODES = 100


def main():
    print("FrozenLake: DQN + classical planner")

    env_seed = ENV_SEED if ENV_SEED is not None else RNG_SEED

    print("--- Training ---")

    dqn_agent = train_dqn(episodes=TRAIN_EPISODES, use_planner=True, rng_seed=RNG_SEED, env_seed=env_seed)
    plot_training_curve(dqn_agent.training_rewards, title="FrozenLake Training Curve - DQN + classical planner")

    print("--- Evaluation ---")

    test_rewards = test_dqn(dqn_agent, episodes=TEST_EPISODES, rng_seed=RNG_SEED, env_seed=env_seed)
    plot_test_distribution(test_rewards, title="FrozenLake Test Distribution - DQN + classical planner")

    print("FrozenLake: DQN with NO classical planner")

    print("--- Training ---")

    dqn_agent = train_dqn(episodes=TRAIN_EPISODES, use_planner=False, rng_seed=RNG_SEED, env_seed=env_seed)
    plot_training_curve(dqn_agent.training_rewards, title="FrozenLake Training Curve - DQN with NO classical planner")

    print("--- Evaluation ---")

    test_rewards = test_dqn(dqn_agent, episodes=TEST_EPISODES, rng_seed=RNG_SEED, env_seed=env_seed)
    plot_test_distribution(test_rewards, title="FrozenLake Test Distribution - DQN with NO classical planner")


if __name__ == "__main__":
    main()
