# ===============================
# Deep Q-Network (DQN) - FrozenLake-v1
# ===============================


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

import planner


def pick_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


ENV_ID = "FrozenLake-v1"
GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY_STEPS = 2000
MEMORY_SIZE = 5000
BATCH_SIZE = 64
WARM_UP_STEPS = 200  # No training until buffer has enough samples
TARGET_UPDATE_STEPS = 100  # Copy policy -> target network periodically

K_PLANNER = 10  # Every K step usa planner


# ===============================
# Q Network
# ===============================
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        hidden = 64
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, x):
        return self.net(x)


# ===============================
# DQN Agent
# ===============================
class DQNAgent:
    def __init__(self, state_size, action_size, device=None):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay_steps = EPSILON_DECAY_STEPS
        self.batch_size = BATCH_SIZE
        self.warm_up_steps = WARM_UP_STEPS
        self.target_update_steps = TARGET_UPDATE_STEPS

        self.device = device if device is not None else pick_device()

        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=MEMORY_SIZE)
        self.steps_done = 0
        self.learn_steps = 0
        self.planner_calls = 0
        self.planner_misses = 0

    def one_hot(self, state):
        vec = np.zeros(self.state_size, dtype=np.float32)
        vec[state] = 1.0
        return vec

    def planner_action(self, env, state):
        self.planner_calls += 1
        problem = planner.define_problem(env, state)
        plan = planner.build_plan(problem)

        if plan is not None:
            # E.g.: "move_0_1_2" → 2
            return int(plan[0].split("_")[-1])

        self.planner_misses += 1
        return plan

    def select_action(self, env, state):
        # planner-guided step
        if self.steps_done % K_PLANNER == 0:
            planned_action = self.planner_action(env, state)
            if planned_action is not None:
                return planned_action

        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_t = torch.tensor(self.one_hot(state)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def store(self, transition):
        self.memory.append(transition)

    def update_epsilon(self):
        frac = min(1.0, self.steps_done / self.epsilon_decay_steps)
        self.epsilon = max(
            self.epsilon_min,
            EPSILON_START - (EPSILON_START - self.epsilon_min) * frac,
            )

    def train_step(self):
        if len(self.memory) < self.batch_size or self.steps_done < self.warm_up_steps:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            np.array([self.one_hot(s) for s in states])
        ).to(self.device)

        next_states = torch.tensor(
            np.array([self.one_hot(s) for s in next_states])
        ).to(self.device)

        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.model(states).gather(1, actions).squeeze()

        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]

        target = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())


# ===============================
# TRAIN
# ===============================
def _win_rate(reward_list):
    if not reward_list:
        return 0.0
    return sum(1 for r in reward_list if r > 0) / len(reward_list)


def train_dqn(episodes=300, log_every=50):
    env = gym.make(ENV_ID, is_slippery=False)

    state_size = env.observation_space.n
    action_size = env.action_space.n

    device = pick_device()
    agent = DQNAgent(state_size, action_size, device=device)

    print(
        "[train] start | "
        f"device={device} | "
        f"lr={LR} gamma={GAMMA} k_planner={K_PLANNER} | "
        f"batch={BATCH_SIZE} warm_up={WARM_UP_STEPS} target_update={TARGET_UPDATE_STEPS} | "
        f"episodes={episodes}"
    )

    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(env, state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store((state, action, reward, next_state, done))

            agent.steps_done += 1
            agent.update_epsilon()
            agent.train_step()

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        if ep % log_every == 0 or ep == episodes - 1:
            tail = rewards[-50:]
            win_tail = 100 * _win_rate(tail)
            print(
                "[train] "
                f"ep {ep}/{episodes - 1} | "
                f"last_reward={total_reward:.0f} | "
                f"mean{len(tail)}={sum(tail) / len(tail):.3f} | "
                f"win_last{len(tail)}={win_tail:.0f}% | "
                f"epsilon={agent.epsilon:.3f} | "
                f"steps={agent.steps_done} | "
                f"learn={agent.learn_steps}"
            )

    wins = sum(1 for r in rewards if r > 0)
    env.close()

    last_n = min(50, episodes)
    tail_final = rewards[-last_n:]
    win_last_n = 100 * _win_rate(tail_final)

    pm = agent.planner_misses
    pc = agent.planner_calls
    miss_pct = 100.0 * pm / pc if pc else 0.0
    print(
        "[train] summary | "
        f"win_rate_all={100 * wins / episodes:.1f}% | "
        f"win_rate_last{last_n}={win_last_n:.1f}% | "
        f"mean_reward={sum(rewards) / len(rewards):.3f} | "
        f"planner_miss={pm}/{pc} ({miss_pct:.1f}%)"
    )

    return agent


# ===============================
# TEST
# ===============================
def test_dqn(agent, episodes=100):
    env = gym.make(ENV_ID, is_slippery=False)
    agent.epsilon = 0.0
    agent.steps_done = 0

    episode_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(env, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            agent.steps_done += 1

        episode_rewards.append(total_reward)

    env.close()

    wins = sum(1 for r in episode_rewards if r > 0)
    mean_r = sum(episode_rewards) / len(episode_rewards)
    last_n = min(50, episodes)
    win_last_n = 100 * _win_rate(episode_rewards[-last_n:])
    print(
        "[test] summary | "
        f"episodes={episodes} | "
        f"success_rate_all={100 * wins / episodes:.1f}% | "
        f"success_rate_last{last_n}={win_last_n:.1f}% | "
        f"mean_reward={mean_r:.3f}"
    )
