"""Choose an experiment by calling one of the functions in main().

Importing this module is safe: training starts only when a function is called.
Budgets are explicit so planned/plain normal training may differ or be zero.
Use dataclasses.replace on a preset to override settings for an experiment.
"""
from configurations import PointMazeOpenConfig, PointMazeMediumConfig
from experiments.comparison import run_planned_vs_plain_dqn
from experiments.warmup_sweep import run_planner_only_warmup_sweep
from reporting.comparison import save_planned_vs_plain_result


def run_open_warmup_sweep(config=None):
    """One Open DQN; pause for greedy evaluation at each cumulative checkpoint."""
    return run_planner_only_warmup_sweep(config or PointMazeOpenConfig())


def run_medium_warmup_sweep(config=None):
    """One Medium DQN; candidate counts and test length come from the preset."""
    return run_planner_only_warmup_sweep(config or PointMazeMediumConfig())


def _run_comparison(config, warmup, planned_training, plain_training, tests):
    result = run_planned_vs_plain_dqn(
        config=config,
        planner_warmup_episodes=warmup,
        planned_train_episodes=planned_training,
        plain_train_episodes=plain_training,
        test_episodes=tests,
    )
    path = save_planned_vs_plain_result(config, result)
    print("Saved comparison: {}".format(path))
    return result, path


def run_open_comparison(config=None, planner_warmup_episodes=25,
                        planned_train_episodes=675, plain_train_episodes=700,
                        test_episodes=5000):
    return _run_comparison(config or PointMazeOpenConfig(), planner_warmup_episodes,
                           planned_train_episodes, plain_train_episodes, test_episodes)


def run_medium_comparison(config=None, planner_warmup_episodes=1000,
                          planned_train_episodes=9000, plain_train_episodes=10000,
                          test_episodes=5000):
    return _run_comparison(config or PointMazeMediumConfig(), planner_warmup_episodes,
                           planned_train_episodes, plain_train_episodes, test_episodes)


def main():
    # Select one experiment here.
    # run_open_warmup_sweep()
    # run_medium_warmup_sweep()
    # run_open_comparison()
    run_medium_comparison()


if __name__ == "__main__":
    main()
