"""Choose an experiment by calling one of the functions in main().

Importing this module is safe: training starts only when a function is called.
Budgets are explicit so planned/plain normal training may differ or be zero.
Use dataclasses.replace on a preset to override settings for an experiment.
"""
from configurations import (PointMazeOpenConfig, PointMazeMediumConfig,
                            PointMazeOpenSACConfig, PointMazeMediumSACConfig)
from experiments.comparison import run_planned_vs_plain
from experiments.dqn_multiseed import run_dqn_multiseed
from experiments.sac_multiseed import run_medium_sac_bc_multiseed
from experiments.warmup_sweep import run_planner_only_warmup_sweep
from reporting.comparison import save_planned_vs_plain_result
from reporting.artifacts import new_run_directory, write_json


def run_open_warmup_sweep(config=None):
    """One Open DQN; pause for greedy evaluation at each cumulative checkpoint."""
    return run_planner_only_warmup_sweep(config or PointMazeOpenConfig())


def run_medium_warmup_sweep(config=None):
    """One Medium DQN; candidate counts and test length come from the preset."""
    return run_planner_only_warmup_sweep(config or PointMazeMediumConfig())


def _run_comparison(config, warmup, planned_training, plain_training, tests):
    result = run_planned_vs_plain(
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


def run_open_sac_warmup_sweep(config=None):
    return run_planner_only_warmup_sweep(config or PointMazeOpenSACConfig())


def run_medium_sac_warmup_sweep(config=None):
    return run_planner_only_warmup_sweep(config or PointMazeMediumSACConfig())


def run_open_sac_comparison(config=None, planner_warmup_episodes=25,
                            planned_train_episodes=75, plain_train_episodes=100,
                            test_episodes=500, evaluation_interval_episodes=0,
                            planner_guidance_mode="planner_only",
                            planner_action_noise_initial_std=0.0,
                            behavior_cloning_initial_weight=0.0,
                            behavior_cloning_final_weight=0.0):
    return _run_sac_comparison(config or PointMazeOpenSACConfig(), planner_warmup_episodes,
                               planned_train_episodes, plain_train_episodes, test_episodes,
                               evaluation_interval_episodes, planner_guidance_mode,
                               planner_action_noise_initial_std,
                               behavior_cloning_initial_weight,
                               behavior_cloning_final_weight)


def run_medium_sac_comparison(config=None, planner_warmup_episodes=400,
                              planned_train_episodes=50, plain_train_episodes=0,
                              test_episodes=200, evaluation_interval_episodes=10,
                              planner_guidance_mode="planner_only",
                              planner_action_noise_initial_std=0.0,
                              behavior_cloning_initial_weight=0.0,
                              behavior_cloning_final_weight=0.0):
    return _run_sac_comparison(config or PointMazeMediumSACConfig(), planner_warmup_episodes,
                               planned_train_episodes, plain_train_episodes, test_episodes,
                               evaluation_interval_episodes, planner_guidance_mode,
                               planner_action_noise_initial_std,
                               behavior_cloning_initial_weight,
                               behavior_cloning_final_weight)


def run_medium_sac_bc_stability_study(config=None):
    """Five-seed BC-vs-planner-only study with interaction-matched plain SAC."""
    return run_medium_sac_bc_multiseed(
        config or PointMazeMediumSACConfig(train_every_steps=8),
        seeds=(42, 43, 44, 45, 46),
        planner_warmup_episodes=400,
        planned_train_episodes=50,
        test_episodes=200,
        behavior_cloning_final_weight=0.4,
    )


def run_open_dqn_pointmaze_study(config=None):
    """Five-seed replication of the selected Open PointMaze DQN protocol."""
    return run_dqn_multiseed(
        config or PointMazeOpenConfig(),
        seeds=(42, 43, 44, 45, 46),
        planner_warmup_episodes=25,
        planned_train_episodes=675,
        plain_train_episodes=700,
        test_episodes=5_000,
        study_name="pointmaze_open_dqn",
    )


def run_medium_dqn_pointmaze_study(config=None):
    """Five-seed replication of the Medium PointMaze DQN protocol."""
    return run_dqn_multiseed(
        config or PointMazeMediumConfig(),
        seeds=(42, 43, 44, 45, 46),
        planner_warmup_episodes=1_000,
        planned_train_episodes=9_000,
        plain_train_episodes=10_000,
        test_episodes=5_000,
        study_name="pointmaze_medium_dqn",
    )


def run_pointmaze_suite():
    """Run every configured five-seed PointMaze study."""
    results = {}
    results["open_dqn"] = run_open_dqn_pointmaze_study()
    results["medium_dqn"] = run_medium_dqn_pointmaze_study()
    sac_config = PointMazeMediumSACConfig(train_every_steps=8)
    results["medium_sac_her"] = run_medium_sac_bc_stability_study(sac_config)
    directory = new_run_directory(sac_config, "pointmaze_suite")
    suite_path = write_json(directory / "suite_index.json", {
        "schema_version": 1,
        "description": "Five-seed PointMaze suite; follow aggregate paths for details.",
        "studies": {
            name: {"aggregate": str(path)}
            for name, (_, path) in results.items()
        },
    })
    print("Saved PointMaze-suite index: {}".format(suite_path), flush=True)
    return results, suite_path


def _run_sac_comparison(config, warmup, training, plain_training, tests, interval,
                        planner_guidance_mode="planner_only",
                        planner_action_noise_initial_std=0.0,
                        behavior_cloning_initial_weight=0.0,
                        behavior_cloning_final_weight=0.0):
    result = run_planned_vs_plain(
        config, planner_warmup_episodes=warmup, planned_train_episodes=training,
        plain_train_episodes=plain_training, test_episodes=tests,
        evaluation_interval_episodes=interval,
        planner_guidance_mode=planner_guidance_mode,
        planner_action_noise_initial_std=planner_action_noise_initial_std,
        behavior_cloning_initial_weight=behavior_cloning_initial_weight,
        behavior_cloning_final_weight=behavior_cloning_final_weight)
    path = save_planned_vs_plain_result(config, result)
    print("Saved SAC comparison: {}".format(path), flush=True)
    return result, path



def main():
    """Uncomment one experiment below; leaving all commented runs nothing."""
    # Select one experiment here.
    # run_open_warmup_sweep()
    # run_medium_warmup_sweep()
    # run_open_comparison()
    # run_medium_comparison()
    # run_open_sac_warmup_sweep()
    # run_medium_sac_warmup_sweep()

    # Full reproducible PointMaze suite. For one study only, call one
    # of run_open_dqn_pointmaze_study(), run_medium_dqn_pointmaze_study(), or
    # run_medium_sac_bc_stability_study() instead.
    run_pointmaze_suite()


if __name__ == "__main__":
    main()
