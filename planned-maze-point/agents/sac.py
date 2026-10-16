"""SB3 SAC adapter for externally controlled, episode-counted experiments.

Our runner owns collection so planner actions cannot be replaced by SB3's
random-start collector. SB3 owns all actor, twin-critic, target and alpha
updates. No imitation loss, recovery controller or epsilon schedule is used.
Low-level SB3 integration is pinned to 2.7.1 and covered by regression tests.
"""
from contextlib import contextmanager
from copy import deepcopy
import random
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from environments.common import ACTION_TABLE
from .replay.point_maze_her import EpisodicPointMazeHER
from .sac_behavior_cloning import PlannerDemonstrationBuffer, PlannerRegularizedSAC


class SACAgent:
    def __init__(self, config, env, behavior_cloning=False):
        if not isinstance(env.action_space, spaces.Box) or env.action_space.shape != (2,):
            raise ValueError("SAC requires native continuous PointMaze actions")
        if env.unwrapped.continuing_task or env.unwrapped.reward_type != "sparse":
            raise ValueError("This HER buffer requires sparse episodic PointMaze")
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.action_space = env.action_space
        self.stored_environment_steps = 0
        self.recovery_events = self.recovery_actions = 0
        self.completed_episodes = 0
        self.behavior_cloning_enabled = behavior_cloning
        self.demonstration_buffer = (
            PlannerDemonstrationBuffer(config.replay_capacity, config.seed + 17)
            if behavior_cloning else None
        )
        model_class = PlannerRegularizedSAC if behavior_cloning else SAC
        extra_model_args = (
            {"demonstration_buffer": self.demonstration_buffer} if behavior_cloning else {}
        )
        self.model = model_class(
            "MultiInputPolicy", env, seed=config.seed, device=config.device,
            learning_rate=config.learning_rate, buffer_size=config.replay_capacity,
            learning_starts=config.learning_starts_transitions, batch_size=config.batch_size,
            gamma=config.gamma, tau=config.tau, ent_coef=config.entropy_coefficient,
            target_entropy="auto", train_freq=config.train_every_steps,
            gradient_steps=config.updates_per_training_event,
            policy_kwargs={"net_arch": list(config.hidden_sizes)},
            replay_buffer_class=EpisodicPointMazeHER,
            replay_buffer_kwargs={"n_sampled_goal": config.her_sampled_goals,
                                  "goal_selection_strategy": config.her_strategy,
                                  "handle_timeout_termination": True},
            **extra_model_args,
        )
        # train() is used without learn(): initialize its logger explicitly.
        self.model.set_logger(configure(folder=None, format_strings=[]))

    @property
    def updates(self):
        return self.model._n_updates

    def reset_episode(self):
        pass

    def planner_action(self, index, noise_std=0.0):
        """Convert grid action to continuous force, optionally perturbing it.

        Planner-directed noise preserves the planner's intended vector as the
        action centre. It is not an SAC action and therefore never causes an
        early handover to an untrained policy.
        """
        action = ACTION_TABLE[index].copy()
        if noise_std:
            action += self.rng.normal(0.0, noise_std, size=action.shape).astype(np.float32)
            action = np.clip(action, self.action_space.low, self.action_space.high)
        return action.astype(np.float32, copy=False)

    def use_planner(self, probability):
        """Sample the external guidance decision from this agent's seeded RNG."""
        return bool(self.rng.random() < probability)

    def select_action(self, observation, training, epsilon=0.0):
        if training and self.stored_environment_steps < self.config.learning_starts_transitions:
            return self.rng.uniform(self.action_space.low, self.action_space.high).astype(np.float32)
        return self.model.predict(observation, deterministic=not training)[0]

    def observe(self, observation, action, reward, next_observation, terminated, truncated, info, config,
                is_planner_action=False):
        def batch(obs):
            return {k: np.asarray(v, dtype=np.float32)[None] for k, v in obs.items()}
        # Replay stores the action ACTUALLY executed, in SB3's normalized space.
        # Store the real final observation, not an auto-reset observation.
        transition_info = dict(info, **{"TimeLimit.truncated": bool(truncated and not terminated)})
        normalized_action = self.model.policy.scale_action(np.asarray(action))
        self.model.replay_buffer.add(
            batch(observation), batch(next_observation),
            normalized_action[None],
            np.array([reward], dtype=np.float32), np.array([terminated or truncated]),
            [transition_info],
        )
        if self.behavior_cloning_enabled and is_planner_action:
            self.demonstration_buffer.add(observation, normalized_action)
        self.stored_environment_steps += 1
        self.model.num_timesteps = self.stored_environment_steps
        self.completed_episodes += int(terminated or truncated)
        # Future HER can only sample completed trajectories. Learning readiness
        # must never change the source of planner warm-up actions.
        if (self.completed_episodes and
                self.stored_environment_steps >= config.learning_starts_transitions and
                self.stored_environment_steps % config.train_every_steps == 0):
            self.model.train(config.updates_per_training_event, config.batch_size)

    def set_behavior_cloning_weight(self, weight):
        if not self.behavior_cloning_enabled:
            if weight:
                raise ValueError("Behaviour cloning was not enabled when this SAC agent was created")
            return
        self.model.set_behavior_cloning_weight(weight)

    def diagnostics(self):
        # SB3 creates log_ent_coef only for adaptive alpha. With a fixed alpha
        # it instead keeps the configured scalar in ent_coef_tensor; reports
        # should expose the actual value in both cases.
        entropy = (
            self.model.log_ent_coef.detach().exp()
            if self.model.log_ent_coef is not None
            else self.model.ent_coef_tensor
        )
        diagnostics = {"entropy_coefficient": float(entropy.detach().cpu()),
                       "optimizer_updates": self.updates}
        if self.behavior_cloning_enabled:
            diagnostics.update(
                behavior_cloning_weight=self.model.behavior_cloning_weight,
                behavior_cloning_loss=self.model.last_behavior_cloning_loss,
                planner_demonstrations=len(self.demonstration_buffer),
            )
        return diagnostics

    @contextmanager
    def evaluation_context(self):
        """No model/replay copies: tests verify that inference never mutates them."""
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        torch_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        own_rng = deepcopy(self.rng.bit_generator.state)
        modes = [(module, module.training) for module in self.model.policy.modules()]
        try:
            self.model.policy.set_training_mode(False)
            yield
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            self.rng.bit_generator.state = own_rng
            for module, mode in modes:
                module.training = mode
