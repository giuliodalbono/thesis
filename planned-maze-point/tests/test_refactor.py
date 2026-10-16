"""Behavioral checks for the architecture and reporting refactor.

Run from planned-maze-point: python -m unittest discover -s tests -v
Uses tiny budgets and temporary output directories, never full experiments.
"""
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from agents import create_agent
from configurations import PointMazeOpenConfig, PointMazeMediumConfig
from environments.common import flatten_observation
from planners import create_planner
from experiments.runner import run_episode
from experiments.comparison import run_planned_vs_plain_dqn
from experiments.warmup_sweep import (
    _evaluate_without_learning, run_planner_only_warmup_sweep,
)
from reporting.comparison import (
    build_comparison_figure, trailing_mean, save_planned_vs_plain_result,
)


class RefactorTests(unittest.TestCase):
    def small_config(self, cls=PointMazeOpenConfig, **overrides):
        values = dict(max_episode_steps=32, batch_size=4,
                      learning_starts_transitions=4, target_sync_updates=2,
                      evaluation_episodes=2, warmup_episode_counts=(1, 2, 4))
        values.update(overrides)
        return replace(cls(), **values)

    def test_evaluation_pause_preserves_continued_learning(self):
        config = self.small_config(stuck_detection_steps=1)

        def train(pause):
            env = config.environment.create(config.max_episode_steps)
            try:
                obs, _ = env.reset(seed=config.seed)
                agent = create_agent(config, flatten_observation(obs).size, env.action_space.n)
                planner = create_planner(config)
                run_episode(env, agent, 42, planner, True, config)
                if pause:
                    before = copy.deepcopy(agent.optimizer.state_dict())
                    weights = {k: v.clone() for k, v in agent.online.state_dict().items()}
                    replay = list(agent.memory)
                    _evaluate_without_learning(env, agent, config)
                    self.assertEqual(agent.optimizer.state_dict()["param_groups"], before["param_groups"])
                    for param, values in before["state"].items():
                        for key, expected in values.items():
                            actual = agent.optimizer.state_dict()["state"][param][key]
                            if torch.is_tensor(expected):
                                self.assertTrue(torch.equal(expected, actual))
                            else:
                                self.assertEqual(expected, actual)
                    for key, value in weights.items():
                        self.assertTrue(torch.equal(value, agent.online.state_dict()[key]))
                    self.assertEqual(len(replay), len(agent.memory))
                    self.assertTrue(all(a is b for a, b in zip(replay, agent.memory)))
                run_episode(env, agent, 43, planner, True, config)
                return agent
            finally:
                env.close()

        continuous = train(False)
        paused = train(True)
        self.assertGreater(paused.updates, 0)
        self.assertEqual(continuous.updates, paused.updates)
        self.assertEqual(continuous.rng.getstate(), paused.rng.getstate())
        for name in ("online", "target"):
            for key, value in getattr(continuous, name).state_dict().items():
                self.assertTrue(torch.equal(value, getattr(paused, name).state_dict()[key]))

    def test_sweep_uses_one_agent_and_writes_only_one_compact_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.small_config(output_dir=Path(tmp))
            with patch("experiments.warmup_sweep.create_agent", wraps=create_agent) as factory:
                report, path = run_planner_only_warmup_sweep(config)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(list(Path(tmp).rglob("*.json")), [path])
            rows = report["checkpoints"]
            self.assertEqual([r["additional_warmup_episodes"] for r in rows], [1, 1, 2])
            self.assertEqual([r["warmup_summary"]["episodes"] for r in rows], [1, 2, 4])
            self.assertTrue(all(r["evaluation_summary"]["episodes"] == 2 for r in rows))
            self.assertNotIn("evaluation", rows[0])
            self.assertNotIn("warmup_segment", rows[0])
            self.assertEqual(json.loads(path.read_text())["configuration"]["agent_name"], "dqn")

    def test_combined_plot_alignment_smoothing_and_success_box(self):
        def episodes(n):
            return [{"reward": float(i % 2)} for i in range(n)]
        result = {
            "config": PointMazeOpenConfig().to_dict(), "test_episodes": 5,
            "planned": {
                "warmup": episodes(3), "training": episodes(400),
                "post_warmup_test_summary": {"success_rate": .2},
                "test_summary": {"success_rate": .8}},
            "plain": {"training": episodes(600), "test_summary": {"success_rate": .6}},
        }
        figure = build_comparison_figure(result)
        rewards, epsilons = figure.axes
        curves = {line.get_label(): line for line in rewards.lines}
        self.assertEqual(curves["Plain DQN reward"].get_xdata()[0], 4)
        self.assertEqual(curves["Planned DQN reward"].get_xdata()[0], 4)
        self.assertEqual(curves["Plain DQN reward"].get_xdata()[-1], 603)
        eps = {line.get_label(): line for line in epsilons.lines}
        self.assertAlmostEqual(eps["Planned DQN epsilon"].get_ydata()[299], .05)
        self.assertGreater(eps["Plain DQN epsilon"].get_ydata()[299], .05)
        self.assertAlmostEqual(trailing_mean([1] * 200 + [0])[200], 199 / 200)
        self.assertEqual(trailing_mean([1])[0], 1)
        box = figure.texts[0].get_text()
        for value in ("20.00%", "80.00%", "60.00%"):
            self.assertIn(value, box)

    def test_both_environments_and_zero_training_reports(self):
        for cls in (PointMazeOpenConfig, PointMazeMediumConfig):
            with tempfile.TemporaryDirectory() as tmp:
                config = self.small_config(cls, output_dir=Path(tmp))
                result = run_planned_vs_plain_dqn(config, 1, 0, 0, 1)
                path1 = save_planned_vs_plain_result(config, result)
                path2 = save_planned_vs_plain_result(config, result)
                self.assertNotEqual(path1, path2)
                self.assertTrue(path1.is_file() and path2.is_file())
                self.assertTrue(Path(result["artifacts"]["comparison_curve"]).is_file())
                self.assertEqual(result["config"]["planner_name"], config.planner_name)
                figure = build_comparison_figure(result)
                self.assertIn("N/A (phase not run)", figure.texts[0].get_text())
                zero = run_planned_vs_plain_dqn(config, 0, 0, 0, 1)
                self.assertEqual(len(build_comparison_figure(zero).axes), 2)

    def test_medium_grid_matches_installed_environment(self):
        config = PointMazeMediumConfig()
        env = config.environment.create(config.max_episode_steps)
        try:
            actual = env.unwrapped.maze
            grid = config.environment.create_grid()
            self.assertEqual([list(row) for row in grid.maze_map], actual.maze_map)
            for row in range(grid.map_length):
                for col in range(grid.map_width):
                    cell = (row, col)
                    np.testing.assert_allclose(grid.cell_centre(cell), actual.cell_rowcol_to_xy(np.array(cell)))
                    if grid.is_free(cell):
                        self.assertEqual(grid.nearest_cell(grid.cell_centre(cell)), cell)
            self.assertNotIn((3, 3), set(grid.neighbours((2, 2))))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
