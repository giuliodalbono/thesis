"""Tiny integration runs: verify plumbing, never interpret as performance."""
from dataclasses import replace
from configurations import PointMazeLargeConfig, PointMazeLargeSACConfig
from .comparison import run_planned_vs_plain
from .warmup_sweep import run_planner_only_warmup_sweep
from reporting.comparison import save_planned_vs_plain_result
from reporting.artifacts import new_run_directory, write_json


def run_large_smoke_tests(output_dir=None):
    """Exercise both agents, BC, sweeps and plots with actual optimizer updates.

    Episode limits and networks are reduced ONLY for these integration checks.
    Planner quality is measured separately with full-length validation episodes.
    """
    import torch
    dqn = replace(PointMazeLargeConfig(), max_episode_steps=32, batch_size=8,
                  learning_starts_transitions=16, target_sync_updates=2,
                  warmup_episode_counts=(1, 2), evaluation_episodes=2)
    sac = replace(PointMazeLargeSACConfig(), max_episode_steps=32, batch_size=8,
                  learning_starts_transitions=16, hidden_sizes=(16, 16),
                  replay_capacity=256, warmup_episode_counts=(1, 2),
                  evaluation_episodes=2)
    if output_dir is not None:
        dqn, sac = replace(dqn, output_dir=output_dir), replace(sac, output_dir=output_dir)
    reports = {}
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for name, config, options in (
            ("dqn", dqn, {}),
            ("sac_her", sac, dict(match_plain_interaction_budget=True)),
            ("sac_her_bc", sac, dict(match_plain_interaction_budget=True,
                planner_guidance_mode="planner_behavior_cloning",
                behavior_cloning_initial_weight=1.0, behavior_cloning_final_weight=0.4)),
        ):
            result = run_planned_vs_plain(config, planner_warmup_episodes=2,
                planned_train_episodes=2, plain_train_episodes=4, test_episodes=2, **options)
            for arm in ("planned", "plain"):
                if result[arm]["optimizer_updates"] < 1:
                    raise AssertionError("Smoke test did not exercise learning: " + name)
                if len(result[arm]["test"]) != 2:
                    raise AssertionError("Missing final evaluations")
            result["purpose"] = "integration_smoke_not_performance"
            path = save_planned_vs_plain_result(config, result)
            reports[name] = str(path)
        for name, config in (("dqn_sweep", dqn), ("sac_sweep", sac)):
            _, path = run_planner_only_warmup_sweep(config)
            reports[name] = str(path)
    finally:
        torch.set_num_threads(previous_threads)
    report = dict(purpose="integration_smoke_not_performance", reports=reports)
    path = write_json(new_run_directory(sac, "smoke") / "summary.json", report)
    print("Large smoke reports: {}".format(path), flush=True)
    return report, path
