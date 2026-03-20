import gymnasium as gym
import planner


def main():
    env = gym.make("FrozenLake-v1", is_slippery=False)

    state, _ = env.reset()

    print(f"Initial State: {state}")

    problem = planner.define_problem(env, state)
    plan = planner.build_plan(problem)
    print(f"\nFirst action to take: {plan[0]}")


if __name__ == "__main__":
    main()
