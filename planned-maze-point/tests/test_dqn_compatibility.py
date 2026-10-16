"""Compare the new adapter runner to the pre-SAC collection/update sequence."""
import unittest
from dataclasses import replace
import numpy as np
import torch
from agents import create_agent
from configurations import PointMazeOpenConfig, PointMazeMediumConfig
from environments.common import flatten_observation, transition_success
from experiments.runner import run_episode
from planners import create_planner


class DQNCompatibilityTests(unittest.TestCase):
    def test_pre_sac_episode_order_and_weights_are_preserved(self):
        for cls in (PointMazeOpenConfig, PointMazeMediumConfig):
            config = replace(cls(), max_episode_steps=32, batch_size=8,
                             learning_starts_transitions=8, target_sync_updates=2,
                             stuck_detection_steps=1)

            def train(reference):
                env = config.environment.create(config.max_episode_steps)
                try:
                    observation, _ = env.reset(seed=config.seed)
                    agent = create_agent(config, flatten_observation(observation).size, env.action_space.n)
                    planner = create_planner(config)
                    history = []
                    for index, (use_planner, learn, epsilon) in enumerate(
                        ((True, True, 0.), (False, True, .7), (False, False, 0.), (False, True, .1))
                    ):
                        seed = config.seed + index
                        selected_planner = planner if use_planner else None
                        if not reference:
                            history.append(run_episode(env, agent, seed, selected_planner, learn, config, epsilon))
                            continue
                        # This is the original runner's action/store/update order,
                        # intentionally using DQN's native API, not its adapter.
                        observation, _ = env.reset(seed=seed)
                        agent.reset_episode()
                        state = flatten_observation(observation)
                        reward_sum, steps, success = 0., 0, False
                        events, actions = agent.recovery_events, agent.recovery_actions
                        terminated = truncated = False
                        while not (terminated or truncated):
                            action = planner.action(observation) if use_planner else agent.act(state, epsilon)
                            next_observation, reward, terminated, truncated, info = env.step(action)
                            next_state = flatten_observation(next_observation)
                            if learn:
                                agent.store((state, action, float(reward), next_state, bool(terminated or truncated)))
                                agent.train_if_due(config.train_every_steps, config.updates_per_training_event)
                            reward_sum += float(reward)
                            steps += 1
                            success = success or transition_success(float(reward), info)
                            observation, state = next_observation, next_state
                        history.append(dict(reward=reward_sum, steps=steps, success=success,
                                            recovery_events=agent.recovery_events - events,
                                            recovery_actions=agent.recovery_actions - actions))
                    return agent, history
                finally:
                    env.close()

            old, old_history = train(True)
            new, new_history = train(False)
            self.assertEqual(old_history, new_history)
            self.assertEqual(old.updates, new.updates)
            self.assertEqual(old.rng.getstate(), new.rng.getstate())
            for name in ("online", "target"):
                for key, expected in getattr(old, name).state_dict().items():
                    self.assertTrue(torch.equal(expected, getattr(new, name).state_dict()[key]))
            for original, adapted in zip(old.memory, new.memory):
                for expected, actual in zip(original, adapted):
                    np.testing.assert_array_equal(expected, actual)
