"""Choose an experiment by calling one of the functions in main().

Importing this module is safe: training starts only when a function is called.
Budgets are explicit so planned/plain normal training may differ or be zero.
Use dataclasses.replace on a preset to override settings for an experiment.
"""
from configurations import (PointMazeOpenConfig, PointMazeMediumConfig,
                            PointMazeOpenSACConfig, PointMazeMediumSACConfig,
                            PointMazeLargeConfig, PointMazeLargeSACConfig)
from experiments.comparison import run_planned_vs_plain
from experiments.dqn_multiseed import run_dqn_multiseed
from experiments.sac_multiseed import run_medium_sac_bc_multiseed
from experiments.sac_multiseed import run_sac_bc_multiseed
from experiments.validation import run_planner_validation, save_planner_validation
from experiments.smoke import run_large_smoke_tests
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


def run_large_planner_validation(config=None, episodes=100):
    """Validate Large steering in the real environment; save without learning."""
    config = config or PointMazeLargeConfig()
    result = run_planner_validation(config, episodes)
    path = save_planner_validation(config, result)
    print("Large planner: {:.1%} success. Report: {}".format(
        result["summary"]["success_rate"], path), flush=True)
    return result, path


def run_large_warmup_sweep(config=None):
    return run_planner_only_warmup_sweep(config or PointMazeLargeConfig())


def run_large_sac_warmup_sweep(config=None):
    """Planner-only SAC+HER sweep, without BC, like the Medium sweep."""
    return run_planner_only_warmup_sweep(config or PointMazeLargeSACConfig())


def run_large_comparison(config=None, planner_warmup_episodes=100,
                         planned_train_episodes=900, plain_train_episodes=1000,
                         test_episodes=500):
    """Small Large DQN pilot; budgets are deliberately not tuned defaults."""
    return _run_comparison(config or PointMazeLargeConfig(), planner_warmup_episodes,
                           planned_train_episodes, plain_train_episodes, test_episodes)


def run_large_sac_comparison(config=None, **budgets):
    config = config or PointMazeLargeSACConfig(seed=42)
    options = dict(planner_warmup_episodes=400, planned_train_episodes=50,
                   plain_train_episodes=0, test_episodes=500,
                   evaluation_interval_episodes=0,
                   planner_guidance_mode="planner_only",
                   behavior_cloning_initial_weight=0.0,
                   behavior_cloning_final_weight=0.0,
                   match_plain_interaction_budget=True)
    options.update(budgets)
    result = run_planned_vs_plain(config, **options)
    path = save_planned_vs_plain_result(config, result)
    print("Saved Large SAC comparison: {}".format(path), flush=True)
    return result, path


def run_large_dqn_study(config=None, **budgets):
    """Multi-seed pilot; override budgets after the Large tuning phase."""
    options = dict(planner_warmup_episodes=25, planned_train_episodes=25,
                   plain_train_episodes=50, test_episodes=100,
                   study_name="pointmaze_large_dqn")
    options.update(budgets)
    return run_dqn_multiseed(config or PointMazeLargeConfig(), **options)


def run_large_sac_bc_comparison(config=None, planner_warmup_episodes=400,
                                planned_train_episodes=50, test_episodes=500,
                                behavior_cloning_initial_weight=1.0,
                                behavior_cloning_final_weight=0.4,
                                evaluation_interval_episodes=0):
    return run_large_sac_comparison(
        config=config, planner_warmup_episodes=planner_warmup_episodes,
        planned_train_episodes=planned_train_episodes, test_episodes=test_episodes,
        planner_guidance_mode="planner_behavior_cloning",
        behavior_cloning_initial_weight=behavior_cloning_initial_weight,
        behavior_cloning_final_weight=behavior_cloning_final_weight,
        evaluation_interval_episodes=evaluation_interval_episodes,
        match_plain_interaction_budget=True, plain_train_episodes=0)


def run_large_sac_bc_stability_study(config=None, seeds=(47, 48, 49, 50, 51),
                                    planner_warmup_episodes=400,
                                    planned_train_episodes=50, test_episodes=5000,
                                    behavior_cloning_initial_weight=1.0,
                                    behavior_cloning_final_weight=0.4,
                                    include_planner_only_control=False):
    return run_sac_bc_multiseed(
        config or PointMazeLargeSACConfig(seed=42), seeds=seeds,
        planner_warmup_episodes=planner_warmup_episodes,
        planned_train_episodes=planned_train_episodes, test_episodes=test_episodes,
        behavior_cloning_initial_weight=behavior_cloning_initial_weight,
        behavior_cloning_final_weight=behavior_cloning_final_weight,
        include_planner_only_control=include_planner_only_control)


def run_large_sac_bc_study(config=None, **budgets):
    """Reuse Medium's study design; its default budgets are a transfer pilot."""
    return run_sac_bc_multiseed(config or PointMazeLargeSACConfig(), **budgets)


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
    """Run the established Open/Medium suite; Large is separate until tuned."""
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
    """Select a service; currently tune Large SAC+HER+BC vs plain SAC+HER."""
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
    # run_pointmaze_suite()
    # run_large_smoke_tests()
    # run_large_warmup_sweep()
    # run_large_planner_validation()
    # run_large_comparison()
    # run_large_sac_warmup_sweep()
    # run_large_sac_comparison()  # Planner-only/no-BC control.
    # Later: pass the chosen parameters to run_large_sac_bc_stability_study().
    # First tuning candidate, seed 42; plain SAC+HER has no BC or planner access.
    # run_large_sac_bc_comparison(
    #     planner_warmup_episodes=400,
    #     planned_train_episodes=50,
    #     test_episodes=200,
    #     behavior_cloning_initial_weight=1.0,
    #     behavior_cloning_final_weight=0.4,
    # )
    run_large_sac_bc_stability_study()


if __name__ == "__main__":
    main()
