"""Tiny real-environment integration tests; no full training required."""
import copy
import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from agents import create_agent
from agents.execution import agent_execution
from agents.evaluation import preserve_training_state
from configurations import (PointMazeOpenSACConfig, PointMazeMediumSACConfig,
                            PointMazeOpenConfig, config_from_dict)
from experiments.runner import run_episode
from experiments.sac_comparison import (run_planned_vs_plain_sac, evaluate_sac,
                                        warmup_planner_probability,
                                        warmup_planner_noise_std,
                                        behavior_cloning_weight)
from experiments.warmup_sweep import run_planner_only_warmup_sweep
from planners import create_planner
from reporting.comparison import build_comparison_figure, save_planned_vs_plain_result
from reporting.sac import build_sac_interaction_figure


class SACTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self, cls=PointMazeOpenSACConfig, **changes):
        values = dict(max_episode_steps=32, learning_starts_transitions=16,
                      batch_size=8, replay_capacity=256, hidden_sizes=(16, 16),
                      warmup_episode_counts=(1, 2, 3), evaluation_episodes=2)
        values.update(changes)
        return replace(cls(), **values)

    def assert_tree_equal(self, a, b):
        if torch.is_tensor(a):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a:
                self.assert_tree_equal(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_tree_equal(x, y)
        else:
            self.assertEqual(a, b)

    def snapshot(self, agent):
        buffer = agent.model.replay_buffer
        return copy.deepcopy({
            "parameters": agent.model.get_parameters(),
            "alpha": agent.model.log_ent_coef,
            "steps": agent.stored_environment_steps, "updates": agent.updates,
            "completed": agent.completed_episodes,
            "replay": {key: value for key, value in vars(buffer).items()
                       if isinstance(value, (np.ndarray, int, bool)) or
                       key in ("observations", "next_observations")},
            "rng": agent.rng.bit_generator.state,
        })

    def test_native_action_spaces_and_config_roundtrip(self):
        for cls in (PointMazeOpenSACConfig, PointMazeMediumSACConfig):
            config = self.config(cls)
            config.validate()
            saved = json.loads(json.dumps(config.to_dict()))
            self.assertEqual(config_from_dict(saved), config)
            env = config.environment.create(32, continuous=True)
            try:
                self.assertEqual(env.action_space.shape, (2,))
                agent = create_agent(config, env=env)
                obs, _ = env.reset(seed=42)
                self.assertTrue(env.action_space.contains(agent.select_action(obs, True)))
                self.assertEqual(agent.recovery_actions, 0)
            finally:
                env.close()
        dqn_env = PointMazeOpenConfig().environment.create(32)
        try:
            self.assertEqual(dqn_env.action_space.n, 8)
        finally:
            dqn_env.close()

    def test_entropy_configuration_accepts_fixed_or_automatic_alpha(self):
        self.config(entropy_coefficient=0.1).validate()
        self.config(entropy_coefficient="auto").validate()
        for invalid in (0, -0.1, "auto_0.1", "invalid", True):
            with self.assertRaises(ValueError):
                self.config(entropy_coefficient=invalid).validate()

    def test_diagnostics_support_fixed_and_automatic_entropy(self):
        for entropy in (0.1, "auto"):
            config = self.config(entropy_coefficient=entropy)
            env = config.environment.create(32, continuous=True)
            try:
                agent = create_agent(config, env=env)
                self.assertGreater(agent.diagnostics()["entropy_coefficient"], 0)
            finally:
                env.close()

    def test_planner_exclusive_collection_and_updates(self):
        config = self.config()
        env = config.environment.create(32, continuous=True)
        try:
            agent = create_agent(config, env=env)
            planner = create_planner(config)
            with patch.object(agent, "select_action", side_effect=AssertionError("Planner bypassed")):
                with patch.object(env, "step", wraps=env.step) as step:
                    run_episode(env, agent, 42, planner, True, config)
            self.assertGreater(agent.updates, 0)
            actions = np.asarray([call.args[0] for call in step.call_args_list])
            np.testing.assert_allclose(agent.model.replay_buffer.actions[:len(actions), 0], actions)
            self.assertEqual(agent.stored_environment_steps, len(actions))
            self.assertEqual(agent.recovery_actions, 0)
        finally:
            env.close()

    def test_annealed_guidance_uses_planner_at_start_and_agent_at_end(self):
        self.assertEqual(warmup_planner_probability("linear_warmup_anneal", 0, 5), 1.0)
        self.assertEqual(warmup_planner_probability("linear_warmup_anneal", 4, 5), 0.0)
        self.assertEqual(warmup_planner_probability("linear_warmup_anneal", 2, 5), 0.5)
        with self.assertRaises(ValueError):
            warmup_planner_probability("unknown", 0, 5)

        config = self.config()
        env = config.environment.create(32, continuous=True)
        try:
            agent = create_agent(config, env=env)
            planner = create_planner(config)
            with patch.object(agent, "select_action", side_effect=AssertionError("Planner not used")):
                first = run_episode(env, agent, 42, planner, True, config,
                                    planner_probability=1.0)
            self.assertEqual(first["planner_actions"], first["steps"])
            self.assertEqual(first["agent_actions"], 0)
            with patch.object(planner, "action", side_effect=AssertionError("Planner used")):
                final = run_episode(env, agent, 43, planner, True, config,
                                    planner_probability=0.0)
            self.assertEqual(final["planner_actions"], 0)
            self.assertEqual(final["agent_actions"], final["steps"])
        finally:
            env.close()

    def test_planner_directed_noise_keeps_planner_control_and_anneals(self):
        self.assertEqual(warmup_planner_probability("planner_action_noise", 2, 5), 1.0)
        self.assertAlmostEqual(warmup_planner_noise_std("planner_action_noise", 0, 5, .2), .2)
        self.assertAlmostEqual(warmup_planner_noise_std("planner_action_noise", 4, 5, .2), 0.)
        self.assertEqual(warmup_planner_noise_std("planner_only", 0, 5, .2), 0.)
        with self.assertRaises(ValueError):
            warmup_planner_noise_std("planner_action_noise", 0, 5, -.1)

        config = self.config()
        env = config.environment.create(32, continuous=True)
        try:
            agent = create_agent(config, env=env)
            planner = create_planner(config)
            observation, _ = env.reset(seed=42)
            base = agent.planner_action(planner.action(observation))
            noisy = agent.planner_action(planner.action(observation), noise_std=.2)
            np.testing.assert_allclose(agent.planner_action(planner.action(observation), noise_std=0.), base)
            self.assertFalse(np.array_equal(base, noisy))
            self.assertTrue(env.action_space.contains(noisy))
            with patch.object(agent, "select_action", side_effect=AssertionError("Actor used")):
                episode = run_episode(env, agent, 42, planner, True, config,
                                      planner_probability=1., planner_action_noise_std=.2)
            self.assertEqual(episode["planner_actions"], episode["steps"])
            self.assertEqual(episode["agent_actions"], 0)
            self.assertEqual(episode["planner_action_noise_std"], .2)
        finally:
            env.close()

    def test_behavior_cloning_uses_only_planner_demonstrations_and_decays(self):
        self.assertEqual(behavior_cloning_weight("planner_behavior_cloning", 1., 0., 0, 5), 1.)
        self.assertEqual(behavior_cloning_weight("planner_behavior_cloning", 1., 0., 4, 5), 0.)
        self.assertEqual(behavior_cloning_weight("planner_only", 1., 0., 0, 5), 0.)
        with self.assertRaises(ValueError):
            behavior_cloning_weight("planner_behavior_cloning", -.1, 0., 0, 5)

        config = self.config(learning_starts_transitions=16)
        env = config.environment.create(32, continuous=True)
        try:
            agent = create_agent(config, env=env, behavior_cloning=True)
            planner = create_planner(config)
            agent.set_behavior_cloning_weight(1.)
            episode = run_episode(env, agent, 42, planner, True, config)
            self.assertEqual(len(agent.demonstration_buffer), episode["planner_actions"])
            self.assertGreater(agent.updates, 0)
            self.assertGreaterEqual(agent.diagnostics()["behavior_cloning_loss"], 0.)
            agent.set_behavior_cloning_weight(0.)
            run_episode(env, agent, 43, None, True, config)
            self.assertEqual(len(agent.demonstration_buffer), episode["planner_actions"])
        finally:
            env.close()

    def test_her_relabels_terminal_and_preserves_real_data(self):
        config = self.config()
        env = config.environment.create(32, continuous=True)
        try:
            agent = create_agent(config, env=env)
            buffer = agent.model.replay_buffer
            # Original goal = (0,0); only the second transition originally succeeds.
            for xy, done in (((1., 0.), False), ((0., 0.), True)):
                obs = {"observation": np.zeros((1, 4), dtype=np.float32),
                       "achieved_goal": np.array([xy], dtype=np.float32),
                       "desired_goal": np.zeros((1, 2), dtype=np.float32)}
                buffer.add(obs, obs, np.zeros((1, 2)), np.array([float(done)]),
                           np.array([done]), [{}])
            before = self.snapshot(agent)
            # Relabelling reverses both terminal decisions.
            with patch.object(buffer, "_sample_goals", return_value=np.array([[1., 0.], [9., 9.]])):
                samples = buffer._get_virtual_samples(np.array([0, 1]), np.array([0, 0]))
            np.testing.assert_array_equal(samples.dones.cpu(), [[1.], [0.]])
            np.testing.assert_array_equal(samples.rewards.cpu(), [[1.], [0.]])
            np.testing.assert_array_equal(samples.observations["desired_goal"].cpu(),
                                          samples.next_observations["desired_goal"].cpu())
            self.assert_tree_equal(before, self.snapshot(agent))
            # A timeout is not a true terminal for real replay transitions.
            buffer.add(obs, obs, np.zeros((1, 2)), np.array([0.]), np.array([True]),
                       [{"TimeLimit.truncated": True}])
            samples = buffer._get_real_samples(np.array([2]), np.array([0]))
            self.assertEqual(samples.dones.item(), 0.)
            # But a new goal reached at the time limit IS a virtual terminal.
            with patch.object(buffer, "_sample_goals", return_value=np.array([[0., 0.]])):
                self.assertEqual(buffer._get_virtual_samples(np.array([2]), np.array([0])).dones.item(), 1.)
        finally:
            env.close()

    def test_future_her_uses_completed_same_episode_goals(self):
        config = self.config()
        env = config.environment.create(32, continuous=True)
        try:
            agent = create_agent(config, env=env)
            buffer = agent.model.replay_buffer
            for index in range(8):
                obs = {"observation": np.zeros((1, 4), dtype=np.float32),
                       "achieved_goal": np.array([[index, 0.]], dtype=np.float32),
                       "desired_goal": np.zeros((1, 2), dtype=np.float32)}
                buffer.add(obs, obs, np.zeros((1, 2)), np.array([0.]),
                           np.array([index in (3, 7)]), [{}])
            for _ in range(20):
                goals = buffer._sample_goals(np.arange(8), np.zeros(8, dtype=int))[:, 0]
                self.assertTrue(np.all(goals >= np.arange(8)))
                self.assertTrue(np.all(goals[:4] < 4))
                self.assertTrue(np.all(goals[4:] < 8))
            self.assertAlmostEqual(buffer.her_ratio, .8)
        finally:
            env.close()

    def test_evaluation_is_frozen_and_resume_is_identical(self):
        config = self.config()
        def train(pause):
            env = config.environment.create(32, continuous=True)
            evaluation = config.environment.create(32, continuous=True)
            try:
                agent = create_agent(config, env=env)
                run_episode(env, agent, 42, create_planner(config), True, config)
                if pause:
                    before = self.snapshot(agent)
                    rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
                    evaluate_sac(evaluation, agent, config, 2)
                    self.assert_tree_equal(before, self.snapshot(agent))
                    self.assert_tree_equal(rng, (random.getstate(), np.random.get_state(), torch.get_rng_state()))
                run_episode(env, agent, 43, None, True, config)
                return self.snapshot(agent)
            finally:
                env.close()
                evaluation.close()
        self.assert_tree_equal(train(False), train(True))

    def test_sweep_keeps_one_agent_and_compact_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(output_dir=Path(tmp))
            with patch("experiments.warmup_sweep.create_agent", wraps=create_agent) as factory:
                report, path = run_planner_only_warmup_sweep(config)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual([r["warmup_summary"]["episodes"] for r in report["checkpoints"]], [1, 2, 3])
            self.assertFalse(report["protocol"]["recovery_can_activate_during_evaluation"])
            self.assertTrue(report["protocol"]["her_enabled"])
            self.assertEqual(report["protocol"]["planner_guidance_mode"], "planner_only")
            self.assertFalse(report["protocol"]["behavior_cloning"])
            self.assertFalse(report["protocol"]["evaluation_uses_planner"])
            self.assertEqual(list(Path(tmp).rglob("*.json")), [path])

    def test_comparison_both_envs_and_zero_budgets(self):
        for cls in (PointMazeOpenSACConfig, PointMazeMediumSACConfig):
            with tempfile.TemporaryDirectory() as tmp:
                config = self.config(cls, output_dir=Path(tmp))
                result = run_planned_vs_plain_sac(config, 1, 2, 3, 2, 1)
                self.assertGreater(result["planned"]["optimizer_updates"], 0)
                self.assertGreater(result["plain"]["optimizer_updates"], 0)
                self.assertEqual(len(result["planned"]["evaluation_checkpoints"]), 3)
                fig = build_comparison_figure(result)
                curves = {line.get_label(): line for line in fig.axes[0].lines}
                self.assertEqual(curves["Plain SAC+HER"].get_xdata()[0], 2)
                self.assertEqual(curves["Planned SAC+HER"].get_xdata()[0], 2)
                self.assertIn("alpha", fig.axes[1].get_ylabel())
                steps = build_sac_interaction_figure(result)
                self.assertEqual(steps.axes[0].lines[1].get_xdata()[0], 0)
                path = save_planned_vs_plain_result(config, result)
                self.assertTrue(path.is_file())
                self.assertTrue(all(Path(p).is_file() for p in result["artifacts"].values()))
                annealed = run_planned_vs_plain_sac(
                    config, 3, 0, 0, 1, planner_guidance_mode="linear_warmup_anneal"
                )
                self.assertEqual(annealed["protocol"]["warmup_guidance_mode"], "linear_warmup_anneal")
                self.assertEqual(annealed["planned"]["warmup"][0]["planner_probability"], 1.0)
                self.assertEqual(annealed["planned"]["warmup"][-1]["planner_probability"], 0.0)
                self.assertGreater(annealed["planned"]["warmup_summary"]["planner_actions"], 0)
                self.assertGreater(annealed["planned"]["warmup_summary"]["agent_actions"], 0)
                noisy = run_planned_vs_plain_sac(
                    config, 3, 0, 0, 1, planner_guidance_mode="planner_action_noise",
                    planner_action_noise_initial_std=.2,
                )
                self.assertEqual(noisy["protocol"]["warmup_guidance_mode"], "planner_action_noise")
                self.assertAlmostEqual(
                    noisy["planned"]["warmup"][0]["planner_action_noise_std"], .2
                )
                self.assertEqual(noisy["planned"]["warmup"][-1]["planner_action_noise_std"], 0.)
                self.assertEqual(noisy["planned"]["warmup_summary"]["agent_actions"], 0)
                bc = run_planned_vs_plain_sac(
                    config, 3, 2, 0, 1, planner_guidance_mode="planner_behavior_cloning",
                    behavior_cloning_initial_weight=1., behavior_cloning_final_weight=0.,
                )
                self.assertTrue(bc["protocol"]["behavior_cloning"])
                self.assertGreater(bc["planned"]["warmup"][-1]["planner_demonstrations"], 0)
                self.assertEqual(bc["planned"]["training"][-1]["behavior_cloning_weight"], 0.)
                for warmup, planned, plain in ((0, 0, 0), (1, 0, 1), (0, 1, 0)):
                    zero = run_planned_vs_plain_sac(config, warmup, planned, plain, 1)
                    self.assertEqual(zero["planned"]["training_summary"]["episodes"], planned)
                    self.assertEqual(zero["planned"]["post_training_test"] is None, planned == 0)
                    build_comparison_figure(zero)

    def test_execution_settings_are_restored(self):
        before = torch.get_num_threads()
        with agent_execution(self.config(torch_threads=2)):
            self.assertEqual(torch.get_num_threads(), 2)
        self.assertEqual(torch.get_num_threads(), before)

    def test_updates_wait_for_complete_episode_and_finite_weights(self):
        config = self.config(learning_starts_transitions=1)
        env = config.environment.create(32, continuous=True)
        try:
            agent = create_agent(config, env=env)
            obs, _ = env.reset(seed=42)
            for index in range(32):
                action = np.zeros(2, dtype=np.float32)
                following, reward, terminated, truncated, info = env.step(action)
                agent.observe(obs, action, reward, following, terminated, truncated, info, config)
                if index < 31:
                    self.assertEqual(agent.updates, 0)
                obs = following
            self.assertEqual(agent.updates, 1)
            self.assertTrue(all(torch.isfinite(p).all() for p in agent.model.policy.parameters()))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
