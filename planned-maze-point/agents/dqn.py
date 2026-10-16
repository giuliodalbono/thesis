"""Double DQN with replay, target synchronization and PointMaze recovery.

Recovery assumes the first two observation entries are Cartesian position.
Both experiment services construct this agent through agents.factory.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np
import torch
from torch import nn


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@dataclass(frozen=True)
class DQNHyperparameters:
    gamma: float
    learning_rate: float
    batch_size: int
    replay_capacity: int
    learning_starts_transitions: int
    target_sync_updates: int


@dataclass(frozen=True)
class RecoveryConfig:
    stuck_detection_steps: int
    opposite_recovery_steps: int
    random_recovery_steps: int
    minimum_directional_progress: float


class DQNAgent:
    """Double DQN learner shared by both experiment services.

    Planner demonstrations are ordinary replay transitions: there is no direct
    imitation loss. The policy learns their rewards through the Bellman target.
    Recovery takes precedence over epsilon-greedy actions during agent control.
    """
    def __init__(self, state_size: int, action_vectors: np.ndarray, hp: DQNHyperparameters, recovery: RecoveryConfig, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.hp = hp
        self.action_vectors = np.asarray(action_vectors, dtype=np.float32)
        self.action_size = len(self.action_vectors)
        self.recovery = recovery
        self.rng = random.Random(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online = QNetwork(state_size, self.action_size).to(self.device)
        self.target = QNetwork(state_size, self.action_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=hp.learning_rate)
        self.memory: Deque[Transition] = deque(maxlen=hp.replay_capacity)
        self.updates = 0
        self.stored_environment_steps = 0
        self.recovery_events = 0
        self.recovery_actions = 0
        self.reset_episode()

    def reset_episode(self) -> None:
        self._last_position: Optional[np.ndarray] = None
        self._last_action: Optional[int] = None
        self._previous_action: Optional[int] = None
        self._blocked_action: Optional[int] = None
        self._blocked_steps = 0
        self._recovery_mode: Optional[str] = None
        self._recovery_remaining = 0

    def _opposite_action(self, action: int) -> int:
        desired = -self.action_vectors[action]
        return int(np.argmax(self.action_vectors @ desired))

    def _update_recovery_state(self, state: np.ndarray) -> None:
        position = np.asarray(state[:2], dtype=np.float32)
        if self._last_position is not None and self._last_action is not None and self._recovery_mode is None:
            if self._previous_action is not None and self._last_action != self._previous_action:
                # The policy has selected a new direction, so it is not
                # repeatedly pressing against the same obstacle.
                self._blocked_steps = 0
            else:
                attempted = self.action_vectors[self._last_action]
                attempted_norm = max(float(np.linalg.norm(attempted)), 1e-8)
                progress = float(np.dot(position - self._last_position, attempted / attempted_norm))
                self._blocked_steps = self._blocked_steps + 1 if progress < self.recovery.minimum_directional_progress else 0
                if self._blocked_steps >= self.recovery.stuck_detection_steps:
                    self._blocked_action = self._last_action
                    self._recovery_mode = "opposite"
                    self._recovery_remaining = self.recovery.opposite_recovery_steps
                    self._blocked_steps = 0
                    self.recovery_events += 1
        self._last_position = position.copy()

    def _recovery_action(self) -> Optional[int]:
        if self._recovery_mode is None:
            return None
        if self._recovery_mode == "opposite":
            action = self._opposite_action(self._blocked_action if self._blocked_action is not None else 0)
        else:
            action = self.rng.randrange(self.action_size)
        self._recovery_remaining -= 1
        self.recovery_actions += 1
        if self._recovery_remaining == 0:
            if self._recovery_mode == "opposite":
                self._recovery_mode = "random"
                self._recovery_remaining = self.recovery.random_recovery_steps
            else:
                self._recovery_mode = None
                self._blocked_action = None
        return action

    def act(self, state: np.ndarray, epsilon: float) -> int:
        self._update_recovery_state(state)
        action = self._recovery_action()
        if action is None:
            if self.rng.random() < epsilon:
                action = self.rng.randrange(self.action_size)
            else:
                with torch.no_grad():
                    values = self.online(torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
                action = int(values.argmax(dim=1).item())
        self._previous_action = self._last_action
        self._last_action = action
        return action

    def store(self, transition: Transition) -> None:
        """Only training phases call this; evaluation must never enter replay."""
        self.memory.append(transition)
        self.stored_environment_steps += 1

    def train_if_due(self, train_every_steps: int, updates_per_training_event: int) -> int:
        """Train only at the requested replay-data collection frequency."""
        if self.stored_environment_steps % train_every_steps != 0:
            return 0
        return sum(self.update() for _ in range(updates_per_training_event))

    def update(self) -> bool:
        """One replay minibatch, Huber TD loss and gradient-clipped optimizer step."""
        if len(self.memory) < max(self.hp.batch_size, self.hp.learning_starts_transitions):
            return False
        states, actions, rewards, next_states, dones = zip(*random.sample(self.memory, self.hp.batch_size))
        states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        prediction = self.online(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            # Double DQN: choose with the online network, evaluate with the
            # target network. This avoids standard DQN's max-value bias.
            next_actions = self.online(next_states_t).argmax(dim=1, keepdim=True)
            next_q_values = self.target(next_states_t).gather(1, next_actions).squeeze(1)
            target = rewards_t + self.hp.gamma * next_q_values * (1.0 - dones_t)
        loss = nn.functional.smooth_l1_loss(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.hp.target_sync_updates == 0:
            self.target.load_state_dict(self.online.state_dict())
        return True
