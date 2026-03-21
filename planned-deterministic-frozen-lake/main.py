from dqn_agent import train_dqn, test_dqn


def main():
    print("FrozenLake: DQN + classical planner")

    print("--- Training ---")
    dqn_agent = train_dqn()
    print("--- Evaluation ---")
    test_dqn(dqn_agent)


if __name__ == "__main__":
    main()
