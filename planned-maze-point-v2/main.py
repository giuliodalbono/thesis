from dataclasses import replace

from warmup_sweep import run_medium_planner_only_warmup_sweep
from point_maze_medium_config import PointMazeMediumConfig
from planned_vs_plain_dqn import run_planned_vs_plain_dqn, save_planned_vs_plain_result


def main() -> None:
    # Future experiments can import run_planner_only_warmup_sweep directly,
    # instead of placing their experiment logic in this launcher.
    # run_planner_only_warmup_sweep()

    # config = PointMazeOpenConfig()
    #
    # result = run_planned_vs_plain_dqn(
    #     config,
    #     planner_warmup_episodes=25,
    #     planned_train_episodes=500,
    #     plain_train_episodes=500,
    #     test_episodes=5000)
    # path = save_planned_vs_plain_result(config, result)
    # print("Saved report to: ", path)

    # run_medium_planner_only_warmup_sweep()

    config = PointMazeMediumConfig()

    result = run_planned_vs_plain_dqn(
        config=config,
        planner_warmup_episodes=1_000,
        planned_train_episodes=9_000,
        plain_train_episodes=10_000,
        test_episodes=5000,
    )

    path = save_planned_vs_plain_result(config, result)
    print(f"Saved report to: {path}")

if __name__ == "__main__":
    main()
