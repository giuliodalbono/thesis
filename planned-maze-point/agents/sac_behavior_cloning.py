"""SAC with a planner-demonstration behaviour-cloning actor regularizer.

This intentionally subclasses the pinned Stable-Baselines3 2.7.1 SAC training
loop. Critic targets, HER sampling and entropy handling remain SB3 SAC; the only
algorithmic change is an MSE term between the deterministic actor action and a
planner action sampled from a separate demonstration buffer.
"""
from collections import deque

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class PlannerDemonstrationBuffer:
    """Planner-only state/action pairs, kept separate from HER replay.

    Actions are stored in SAC's normalized [-1, 1] action coordinates, matching
    both the actor output and replay actions. Sampling with replacement is
    deliberate: early training can start before a full actor batch is available.
    """
    def __init__(self, capacity, seed):
        self.data = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, observation, normalized_action):
        self.data.append((
            {key: np.asarray(value, dtype=np.float32).copy()
             for key, value in observation.items()},
            np.asarray(normalized_action, dtype=np.float32).copy(),
        ))

    def __len__(self):
        return len(self.data)

    def sample(self, batch_size, device):
        if not self.data:
            raise RuntimeError("Cannot behaviour-clone without planner demonstrations")
        indices = self.rng.integers(0, len(self.data), size=batch_size)
        rows = [self.data[index] for index in indices]
        keys = rows[0][0].keys()
        observations = {
            key: th.as_tensor(np.stack([row[0][key] for row in rows]), device=device)
            for key in keys
        }
        actions = th.as_tensor(np.stack([row[1] for row in rows]), device=device)
        return observations, actions


class PlannerRegularizedSAC(SAC):
    """SB3 SAC plus ``bc_weight * MSE(actor_mean, planner_action)``."""
    def __init__(self, *args, demonstration_buffer, **kwargs):
        self.demonstration_buffer = demonstration_buffer
        self.behavior_cloning_weight = 0.0
        self.last_behavior_cloning_loss = 0.0
        super().__init__(*args, **kwargs)

    def set_behavior_cloning_weight(self, weight):
        if weight < 0:
            raise ValueError("behavior_cloning_weight must be nonnegative")
        self.behavior_cloning_weight = float(weight)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # This is the SB3 2.7.1 SAC loop with the marked actor-loss addition.
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs, actor_losses, critic_losses, bc_losses = [], [], [], [], []
        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            # Behaviour-cloning addition: compare the actor's deterministic,
            # squashed action to actions actually executed by the planner.
            bc_loss = None
            if self.behavior_cloning_weight > 0.0 and len(self.demonstration_buffer):
                demo_observations, demo_actions = self.demonstration_buffer.sample(batch_size, self.device)
                deterministic_actions = self.actor(demo_observations, deterministic=True)
                bc_loss = F.mse_loss(deterministic_actions, demo_actions)
                actor_loss = actor_loss + self.behavior_cloning_weight * bc_loss
                bc_losses.append(bc_loss.item())

            actor_losses.append(actor_loss.item())
            self.critic.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.last_behavior_cloning_loss = float(np.mean(bc_losses)) if bc_losses else 0.0
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/bc_weight", self.behavior_cloning_weight)
        self.logger.record("train/bc_loss", self.last_behavior_cloning_loss)
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
