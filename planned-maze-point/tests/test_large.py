"""Large geometry and real learning smoke checks, plus actor-gradient regression."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from configurations import (PointMazeLargeConfig, PointMazeLargeSACConfig,
                            config_from_dict)
from planners import create_planner, planner_for_env
from agents import create_agent
from experiments.runner import run_episode
from experiments.smoke import run_large_smoke_tests
from experiments.sac_multiseed import run_sac_bc_multiseed
from experiments.sac_comparison import evaluate_sac


class LargeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_dqn_screening_preset_preserves_learning_settings(self):
        from configurations import PointMazeMediumConfig
        large, medium = PointMazeLargeConfig(), PointMazeMediumConfig()
        large.validate()
        # Checkpoints/evaluation counts are tuning inputs, not fixed contracts.
        self.assertTrue(large.warmup_episode_counts)
        # Isolate the warm-up budget: no simultaneous learning/recovery tuning.
        differences = {"env_id", "max_episode_steps", "warmup_episode_counts",
                       "evaluation_episodes"}
        from dataclasses import fields
        for field in fields(medium):
            if field.name not in differences:
                self.assertEqual(getattr(large, field.name), getattr(medium, field.name))
        self.assertEqual(medium.warmup_episode_counts, (1000, 2000, 5000))
        self.assertEqual(medium.evaluation_episodes, 5000)

    def test_large_sac_screening_entrypoint(self):
        from unittest.mock import patch
        from configurations import PointMazeMediumSACConfig
        import main
        config = PointMazeLargeSACConfig()
        config.validate()
        self.assertEqual(config.warmup_episode_counts, (50, 100, 200, 400, 600, 800, 1000))
        self.assertEqual(config.evaluation_episodes, 200)
        self.assertEqual(config.max_episode_steps, 2000)
        self.assertEqual(config.her_strategy, "future")
        self.assertEqual(config.entropy_coefficient, .1)
        self.assertEqual(config.train_every_steps, 8)
        self.assertEqual(PointMazeMediumSACConfig().evaluation_episodes, 500)
        # The sweep remains callable even when main selects a comparison.
        # Mock only orchestration here; real learning is covered by smoke tests.
        with patch.object(main, "run_planner_only_warmup_sweep") as sweep:
            main.run_large_sac_warmup_sweep()
            sweep.assert_called_once_with(config)

    def test_large_sac_comparison_entrypoint_and_budget_overrides(self):
        from unittest.mock import patch
        import main
        expected = dict(planner_warmup_episodes=400, planned_train_episodes=50,
                        plain_train_episodes=0, test_episodes=500,
                        evaluation_interval_episodes=0,
                        planner_guidance_mode="planner_only",
                        behavior_cloning_initial_weight=0.0,
                        behavior_cloning_final_weight=0.0,
                        match_plain_interaction_budget=True)
        config = PointMazeLargeSACConfig(seed=42)
        with patch.object(main, "run_planned_vs_plain", return_value={}) as run, \
             patch.object(main, "save_planned_vs_plain_result", return_value=Path("report.json")) as save:
            main.run_large_sac_comparison()
            run.assert_called_once_with(config, **expected)
            save.assert_called_once_with(config, {})
            # Fixed-episode and zero-phase callers still override the defaults.
            overrides = dict(planner_warmup_episodes=0, planned_train_episodes=0,
                             plain_train_episodes=3, test_episodes=2,
                             match_plain_interaction_budget=False)
            custom = replace(config, seed=43)
            main.run_large_sac_comparison(config=custom, **overrides)
            expected.update(overrides)
            run.assert_called_with(custom, **expected)

    def test_large_bc_tuning_and_stability_entrypoints(self):
        from unittest.mock import patch
        import main
        config = PointMazeLargeSACConfig(seed=42)
        with patch.object(main, "run_planned_vs_plain", return_value={}) as run, \
             patch.object(main, "save_planned_vs_plain_result", return_value=Path("report.json")):
            main.run_large_sac_bc_comparison()
            args, options = run.call_args
            self.assertEqual(args, (config,))
            self.assertEqual(options["planner_guidance_mode"], "planner_behavior_cloning")
            self.assertEqual(options["planner_warmup_episodes"], 400)
            self.assertEqual(options["planned_train_episodes"], 50)
            self.assertEqual(options["test_episodes"], 500)
            self.assertEqual(options["behavior_cloning_initial_weight"], 1.0)
            self.assertEqual(options["behavior_cloning_final_weight"], .4)
            self.assertTrue(options["match_plain_interaction_budget"])
            custom = replace(config, seed=43, learning_rate=5e-5)
            main.run_large_sac_bc_comparison(custom,
                behavior_cloning_initial_weight=.8, behavior_cloning_final_weight=.2)
            self.assertEqual(run.call_args.args, (custom,))
            self.assertEqual(run.call_args.kwargs["behavior_cloning_initial_weight"], .8)
            self.assertEqual(run.call_args.kwargs["behavior_cloning_final_weight"], .2)
        with patch.object(main, "run_sac_bc_multiseed") as study:
            main.run_large_sac_bc_stability_study()
            self.assertEqual(study.call_args.args, (config,))
            self.assertEqual(study.call_args.kwargs["seeds"], (47, 48, 49, 50, 51))
            self.assertFalse(study.call_args.kwargs["include_planner_only_control"])
            main.run_large_sac_bc_stability_study(custom, seeds=(45, 46),
                behavior_cloning_initial_weight=.8, behavior_cloning_final_weight=.2,
                include_planner_only_control=True)
            self.assertEqual(study.call_args.args, (custom,))
            self.assertEqual(study.call_args.kwargs["behavior_cloning_initial_weight"], .8)
            self.assertEqual(study.call_args.kwargs["behavior_cloning_final_weight"], .2)
            self.assertTrue(study.call_args.kwargs["include_planner_only_control"])

    def test_pointmaze_suite_includes_large_studies(self):
        from unittest.mock import patch
        import main
        results = {name: ({"study": name}, Path(name + ".json")) for name in (
            "open", "medium", "medium_sac", "large", "large_sac")}
        with patch.object(main, "run_open_dqn_pointmaze_study", return_value=results["open"]) as open_dqn, \
             patch.object(main, "run_medium_dqn_pointmaze_study", return_value=results["medium"]) as medium_dqn, \
             patch.object(main, "run_medium_sac_bc_stability_study", return_value=results["medium_sac"]) as medium_sac, \
             patch.object(main, "run_large_dqn_pointmaze_study", return_value=results["large"]) as large_dqn, \
             patch.object(main, "run_large_sac_bc_stability_study", return_value=results["large_sac"]) as large_sac, \
             patch.object(main, "new_run_directory", return_value=Path("suite")), \
             patch.object(main, "write_json", return_value=Path("suite/suite_index.json")) as write:
            suite, path = main.run_pointmaze_suite()
        self.assertEqual(set(suite), {
            "open_dqn", "medium_dqn", "medium_sac_her", "large_dqn", "large_sac_her_bc"})
        self.assertEqual(path, Path("suite/suite_index.json"))
        open_dqn.assert_called_once_with()
        medium_dqn.assert_called_once_with()
        large_dqn.assert_called_once_with()
        large_sac.assert_called_once_with()
        medium_config = medium_sac.call_args.args[0]
        self.assertEqual(medium_config.env_id, "PointMaze_Medium-v3")
        self.assertEqual(medium_config.train_every_steps, 8)
        suite_index = write.call_args.args[1]
        self.assertEqual(set(suite_index["studies"]), set(suite))

    def test_main_is_a_safe_selector(self):
        from unittest.mock import patch
        import main
        with patch.object(main, "run_pointmaze_suite") as suite:
            self.assertIsNone(main.main())
            suite.assert_not_called()

    def test_geometry_routes_and_config_roundtrip(self):
        for cls in (PointMazeLargeConfig, PointMazeLargeSACConfig):
            config = cls()
            config.validate()
            self.assertEqual(config_from_dict(json.loads(json.dumps(config.to_dict()))), config)
            with self.assertRaises(ValueError):
                replace(config, env_id="PointMaze_Medium-v3").validate()
        config = PointMazeLargeConfig()
        env = config.environment.create(config.max_episode_steps)
        try:
            actual, grid = env.unwrapped.maze, config.environment.create_grid()
            self.assertEqual([list(row) for row in grid.maze_map], actual.maze_map)
            self.assertEqual(grid.maze_size_scaling, actual.maze_size_scaling)
            cells = []
            for row in range(grid.map_length):
                for column in range(grid.map_width):
                    cell = (row, column)
                    np.testing.assert_allclose(grid.cell_centre(cell),
                        actual.cell_rowcol_to_xy(np.array(cell)))
                    if grid.is_free(cell):
                        cells.append(cell)
                        self.assertEqual(grid.nearest_cell(grid.cell_centre(cell)), cell)
            planner = create_planner(config)
            self.assertIs(type(planner), type(planner_for_env(config.env_id)))
            # Exhaust all start/goal pairs: no route cycles or corner cutting.
            for goal in cells:
                policy = planner._policy_to(goal)
                for start in cells:
                    current, visited = start, set()
                    while current != goal:
                        self.assertNotIn(current, visited)
                        visited.add(current)
                        nxt = policy[current]
                        self.assertIn(nxt, list(grid.neighbours(current)))
                        dr, dc = nxt[0] - current[0], nxt[1] - current[1]
                        if dr and dc:
                            self.assertTrue(grid.is_free((current[0] + dr, current[1])))
                            self.assertTrue(grid.is_free((current[0], current[1] + dc)))
                        current = nxt
        finally:
            env.close()

    def test_large_reports_sweeps_and_generic_multiseed(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, path = run_large_smoke_tests(Path(tmp))
            self.assertTrue(path.exists())
            for saved in report["reports"].values():
                self.assertTrue(Path(saved).exists())
            bc = json.loads(Path(report["reports"]["sac_her_bc"]).read_text())
            self.assertEqual(bc["planned"]["training"][-1]["behavior_cloning_weight"], .4)
            self.assertTrue(bc["protocol"]["imitation_loss"])
            self.assertEqual(bc["protocol"]["bc_implementation_version"], 2)
            self.assertIsNotNone(bc["plain"]["post_training_test_summary"])
            self.assertGreaterEqual(bc["plain"]["pre_test_environment_steps"],
                                    bc["planned"]["pre_test_environment_steps"])
            for artifact in bc["artifacts"].values():
                self.assertTrue(Path(artifact).is_file())
            config = replace(PointMazeLargeSACConfig(), max_episode_steps=16,
                batch_size=4, learning_starts_transitions=8, hidden_sizes=(8, 8),
                replay_capacity=128, output_dir=Path(tmp))
            aggregate, _ = run_sac_bc_multiseed(config, seeds=(42,),
                planner_warmup_episodes=1, planned_train_episodes=1, test_episodes=1)
            self.assertEqual(len(aggregate["runs"]), 2)
            # The optional BC-only study carries tuned weights through learning
            # and reporting, while the original default still runs both modes.
            bc_only, path = run_sac_bc_multiseed(config, seeds=(42,),
                planner_warmup_episodes=1, planned_train_episodes=2, test_episodes=1,
                behavior_cloning_initial_weight=.8, behavior_cloning_final_weight=.2,
                include_planner_only_control=False)
            self.assertEqual(len(bc_only["runs"]), 1)
            self.assertEqual(bc_only["protocol"]["behavior_cloning_weight"],
                             dict(initial=.8, final=.2))
            self.assertTrue(path.exists())
            detail = json.loads(Path(bc_only["runs"][0]["report"]).read_text())
            self.assertEqual(detail["planned"]["warmup"][-1]["behavior_cloning_weight"], .8)
            self.assertAlmostEqual(detail["planned"]["training"][-1]["behavior_cloning_weight"], .2)
            self.assertEqual(detail["plain"]["warmup_summary"]["episodes"], 0)
            self.assertEqual(detail["plain"]["training_summary"]["planner_actions"], 0)
            self.assertGreaterEqual(detail["plain"]["pre_test_environment_steps"],
                                    detail["planned"]["pre_test_environment_steps"])
            self.assertIn("planned_post_warmup_success", bc_only["runs"][0])

    def test_bc_ignores_stale_actor_gradients_and_evaluation_is_frozen(self):
        # Poison actor gradients before training. Correct zero_grad must make
        # both runs identical; the previous critic.zero_grad typo fails this.
        config = replace(PointMazeLargeSACConfig(), max_episode_steps=32,
            batch_size=8, learning_starts_transitions=5000,
            replay_capacity=256, hidden_sizes=(16, 16))

        def trained(poison, evaluate):
            env = config.environment.create(32, continuous=True)
            evaluation_env = config.environment.create(32, continuous=True)
            try:
                agent = create_agent(config, env=env, behavior_cloning=True)
                agent.set_behavior_cloning_weight(.4)
                planner = create_planner(config)
                for i in range(2):
                    run_episode(env, agent, config.seed + i, planner, True, config)
                if evaluate:
                    before = len(agent.demonstration_buffer)
                    evaluate_sac(evaluation_env, agent, config, 2)
                    self.assertEqual(before, len(agent.demonstration_buffer))
                    self.assertEqual(agent.updates, 0)
                if poison:
                    for parameter in agent.model.actor.parameters():
                        parameter.grad = torch.full_like(parameter, 1000.)
                torch.manual_seed(123)
                agent.model.train(2, 8)
                return {key: value.clone() for key, value in agent.model.policy.state_dict().items()}
            finally:
                env.close()
                evaluation_env.close()

        reference = trained(False, False)
        actual = trained(True, True)
        for key in reference:
            self.assertTrue(torch.isfinite(actual[key]).all())
            self.assertTrue(torch.equal(reference[key], actual[key]), key)
