from __future__ import annotations

import random
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from point_maze_wrappers import (
    ContinuousToDiscreteActionWrapper,
    GoalObservationFlattenWrapper,
    build_action_table,
)
from planner import DeterministicPointMazePlanner, PlannerConfig

try:
    import gymnasium_robotics
except ImportError as exc:
    raise ImportError(
        "gymnasium_robotics is required for PointMaze. "
        "Install it with: pip install gymnasium-robotics"
    ) from exc

gym.register_envs(gymnasium_robotics)


def pick_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


ENV_ID = "PointMaze_Open-v3"
GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_MIN = 0.05
BATCH_SIZE = 64
WARM_UP_STEPS = 1000
TARGET_UPDATE_STEPS = 200
CHECKPOINT_TYPES = {"planner_warmup", "trained"}


def _validate_decay_factor(factor: float) -> None:
    if not 0.0 < factor < 1.0:
        raise ValueError(f"epsilon_decay_factor must be in (0, 1), got {factor}")


def get_checkpoint_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_dir = project_root / "resources" / "planned-point-maze" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def checkpoint_path(
    env_label: str,
    checkpoint_type: str,
    action_table_mode: str,
    checkpoint_dir=None,
) -> Path:
    if checkpoint_type not in CHECKPOINT_TYPES:
        raise ValueError(
            f"Unknown checkpoint_type={checkpoint_type!r}. "
            f"Expected one of {sorted(CHECKPOINT_TYPES)}"
        )
    output_dir = get_checkpoint_dir() if checkpoint_dir is None else Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_env_label = str(env_label).replace(" ", "_")
    safe_mode = str(action_table_mode).replace(" ", "_")
    return output_dir / f"PointMaze_{safe_env_label}_{safe_mode}_dqn_{checkpoint_type}.pt"


def _build_checkpoint_payload(
    agent: "DQNAgent",
    *,
    checkpoint_type: str,
    env_id: str,
    env_label: str,
    action_table_mode: str,
    train_episodes_completed: int,
    training_rewards: List[float],
    training_epsilon: List[float],
    planner_warmup_rewards: List[float],
    agent_eval_warmup_rewards: List[float],
    warmup_test_rewards: List[float],
) -> Dict[str, Any]:
    return {
        "version": 1,
        "checkpoint_type": checkpoint_type,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "env_id": env_id,
        "env_label": env_label,
        "action_table_mode": action_table_mode,
        "train_episodes_completed": int(train_episodes_completed),
        "agent_state": {
            "state_size": int(agent.state_size),
            "action_size": int(agent.action_size),
            "memory_size": int(agent.memory_size),
            "epsilon": float(agent.epsilon),
            "epsilon_start": float(agent.epsilon_start),
            "epsilon_min": float(agent.epsilon_min),
            "epsilon_decay_factor": float(agent.epsilon_decay_factor),
            "steps_done": int(agent.steps_done),
            "learn_steps": int(agent.learn_steps),
            "episodes_since_decay_start": int(agent.episodes_since_decay_start),
            "training_time_seconds": float(agent.training_time_seconds),
            "training_episodes": int(agent.training_episodes),
            "planner_calls": int(agent.planner_calls),
            "planner_misses": int(agent.planner_misses),
            "planner_cache_hits": int(agent.planner_cache_hits),
            "planner_time_seconds": float(agent.planner_time_seconds),
            "planner_grid_size": int(agent.planner_grid_size),
            "planner_warm_up_episodes": int(agent.planner_warm_up_episodes),
            "use_planner_warm_up": bool(agent.use_planner_warm_up),
        },
        "training_metrics": {
            "training_rewards": list(training_rewards),
            "training_epsilon": list(training_epsilon),
            "planner_warmup_rewards": list(planner_warmup_rewards),
            "agent_eval_warmup_rewards": list(agent_eval_warmup_rewards),
            "warmup_test_rewards": list(warmup_test_rewards),
        },
        "model_state_dict": agent.model.state_dict(),
        "target_model_state_dict": agent.target_model.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
    }


def save_agent_checkpoint(
    agent: "DQNAgent",
    path,
    *,
    checkpoint_type: str,
    env_id: str,
    env_label: str,
    action_table_mode: str,
    train_episodes_completed: int,
    training_rewards: List[float],
    training_epsilon: List[float],
    planner_warmup_rewards: List[float],
    agent_eval_warmup_rewards: List[float],
    warmup_test_rewards: List[float],
) -> Path:
    checkpoint_file = Path(path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_checkpoint_payload(
        agent,
        checkpoint_type=checkpoint_type,
        env_id=env_id,
        env_label=env_label,
        action_table_mode=action_table_mode,
        train_episodes_completed=train_episodes_completed,
        training_rewards=training_rewards,
        training_epsilon=training_epsilon,
        planner_warmup_rewards=planner_warmup_rewards,
        agent_eval_warmup_rewards=agent_eval_warmup_rewards,
        warmup_test_rewards=warmup_test_rewards,
    )
    torch.save(payload, checkpoint_file)
    return checkpoint_file


def load_agent_checkpoint(path) -> Tuple["DQNAgent", Dict[str, Any]]:
    checkpoint_file = Path(path)
    payload = torch.load(checkpoint_file, map_location=pick_device())
    agent_state = payload["agent_state"]

    agent = DQNAgent(
        state_size=agent_state["state_size"],
        action_size=agent_state["action_size"],
        epsilon_decay_factor=agent_state["epsilon_decay_factor"],
        memory_size=agent_state["memory_size"],
        device=pick_device(),
    )
    agent.model.load_state_dict(payload["model_state_dict"])
    agent.target_model.load_state_dict(payload["target_model_state_dict"])
    agent.optimizer.load_state_dict(payload["optimizer_state_dict"])

    agent.epsilon = float(agent_state["epsilon"])
    agent.epsilon_start = float(agent_state["epsilon_start"])
    agent.epsilon_min = float(agent_state["epsilon_min"])
    agent.steps_done = int(agent_state["steps_done"])
    agent.learn_steps = int(agent_state["learn_steps"])
    agent.episodes_since_decay_start = int(agent_state["episodes_since_decay_start"])
    agent.training_time_seconds = float(agent_state["training_time_seconds"])
    agent.training_episodes = int(agent_state["training_episodes"])
    agent.planner_calls = int(agent_state["planner_calls"])
    agent.planner_misses = int(agent_state["planner_misses"])
    agent.planner_cache_hits = int(agent_state["planner_cache_hits"])
    agent.planner_time_seconds = float(agent_state["planner_time_seconds"])
    agent.planner_grid_size = int(agent_state["planner_grid_size"])
    agent.planner_warm_up_episodes = int(agent_state["planner_warm_up_episodes"])
    agent.use_planner_warm_up = bool(agent_state["use_planner_warm_up"])

    metrics = payload["training_metrics"]
    agent.training_rewards = list(metrics["training_rewards"])
    agent.training_epsilon = list(metrics["training_epsilon"])
    agent.planner_warmup_rewards = list(metrics["planner_warmup_rewards"])
    agent.agent_eval_warmup_rewards = list(metrics["agent_eval_warmup_rewards"])
    agent.warmup_test_rewards = list(metrics["warmup_test_rewards"])

    metadata = {
        "checkpoint_path": str(checkpoint_file),
        "checkpoint_type": payload.get("checkpoint_type"),
        "env_id": payload.get("env_id", ENV_ID),
        "env_label": payload.get("env_label", "Medium_sparse"),
        "action_table_mode": payload.get("action_table_mode", "nine"),
        "train_episodes_completed": payload.get("train_episodes_completed", 0),
        "saved_at": payload.get("saved_at"),
        "planner_warm_up_episodes": agent_state.get("planner_warm_up_episodes", 0),
        "use_planner_warm_up": agent_state.get("use_planner_warm_up", False),
        "planner_grid_size": agent_state.get("planner_grid_size", 21),
    }
    return agent, metadata


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        hidden = 128
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        epsilon_decay_factor: float,
        memory_size: int,
        device: Optional[torch.device] = None,
        planner: Optional[DeterministicPointMazePlanner] = None,
        planner_grid_size: int = 21,
        planner_warm_up_episodes: int = 0,
        use_planner_warm_up: bool = False,
    ):
        _validate_decay_factor(epsilon_decay_factor)

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = GAMMA
        self.epsilon_start = EPSILON_START
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay_factor = epsilon_decay_factor
        self.batch_size = BATCH_SIZE
        self.warm_up_steps = WARM_UP_STEPS
        self.target_update_steps = TARGET_UPDATE_STEPS
        self.memory_size = memory_size
        self.training_time_seconds = 0.0
        self.training_episodes = 0
        self.episodes_since_decay_start = 0
        self.planner = planner
        self.planner_grid_size = int(planner_grid_size)
        self.planner_warm_up_episodes = int(planner_warm_up_episodes)
        self.use_planner_warm_up = bool(use_planner_warm_up)
        self.planner_calls = 0
        self.planner_misses = 0
        self.planner_cache_hits = 0
        self.planner_time_seconds = 0.0

        self.device = device if device is not None else pick_device()
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = deque(maxlen=memory_size)
        self.steps_done = 0
        self.learn_steps = 0
        self.training_rewards: List[float] = []
        self.training_epsilon: List[float] = []
        self.planner_warmup_rewards: List[float] = []
        self.agent_eval_warmup_rewards: List[float] = []
        self.warmup_test_rewards: List[float] = []

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.model(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def store(self, transition) -> None:
        self.memory.append(transition)

    def update_epsilon(self, episode_idx: int) -> None:
        self.episodes_since_decay_start = int(episode_idx)
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_start * (self.epsilon_decay_factor ** episode_idx),
        )

    def select_planned_action(self, state: np.ndarray) -> int:
        if self.planner is None:
            self.planner_misses += 1
            return random.randrange(self.action_size)

        t0 = time.perf_counter()
        planned_action = self.planner.first_action(state)
        self.planner_time_seconds += time.perf_counter() - t0
        self.planner_calls += 1
        self.planner_cache_hits = int(getattr(self.planner, "cache_hits", self.planner_cache_hits))
        if planned_action is None:
            self.planner_misses += 1
            self.planner_misses = max(
                self.planner_misses, int(getattr(self.planner, "misses", self.planner_misses))
            )
            return random.randrange(self.action_size)
        if not 0 <= int(planned_action) < self.action_size:
            self.planner_misses += 1
            return random.randrange(self.action_size)
        return int(planned_action)

    def train_step(self) -> None:
        if len(self.memory) < self.batch_size or self.steps_done < self.warm_up_steps:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.model(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_q = self.target_model(next_states_t).max(1)[0]
        targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())


def _success_rate(reward_list: List[float]) -> float:
    if not reward_list:
        return 0.0
    return sum(1 for r in reward_list if r > 0.0) / len(reward_list)


def _evaluation_episode_seed(base_seed: Optional[int], episode_idx: int):
    return None if base_seed is None else base_seed + episode_idx


def _make_env(
    env_id: str = ENV_ID,
    action_table_mode: str = "nine",
    max_episode_steps: Optional[int] = None,
) -> gym.Env:
    kwargs = {"continuing_task": False}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = int(max_episode_steps)
    base_env = gym.make(env_id, render_mode="human", **kwargs)
    env = GoalObservationFlattenWrapper(base_env)
    env = ContinuousToDiscreteActionWrapper(env, build_action_table(action_table_mode))
    return env


def train_dqn(
    episodes: int,
    rng_seed: int,
    env_seed: Optional[int],
    env_id: str = ENV_ID,
    env_label: str = "Medium_sparse",
    action_table_mode: str = "nine",
    epsilon_decay_factor: float = 0.9995,
    memory_size: int = 200_000,
    checkpoint_dir=None,
    max_episode_steps: Optional[int] = None,
    use_planner_warm_up: bool = True,
    planner_warm_up_episodes: int = 1500,
    planner_grid_size: int = 21,
    train_every_steps: int = 4,
    enable_alternating_warmup: bool = False,
    warmup_test_episodes: int = 0,
) -> DQNAgent:
    set_seed(rng_seed)
    env_base_seed = env_seed
    if train_every_steps < 1:
        raise ValueError(f"train_every_steps must be >= 1, got {train_every_steps}")

    env = _make_env(
        env_id=env_id,
        action_table_mode=action_table_mode,
        max_episode_steps=max_episode_steps,
    )
    state_size = int(np.prod(env.observation_space.shape))
    action_size = env.action_space.n
    action_table = build_action_table(action_table_mode)
    planner = DeterministicPointMazePlanner(
        action_table=action_table,
        config=PlannerConfig(grid_size=int(planner_grid_size)),
    )
    device = pick_device()
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        epsilon_decay_factor=epsilon_decay_factor,
        memory_size=memory_size,
        device=device,
        planner=planner,
        planner_grid_size=planner_grid_size,
        planner_warm_up_episodes=planner_warm_up_episodes,
        use_planner_warm_up=use_planner_warm_up,
    )

    checkpoint_root = None if checkpoint_dir is None else Path(checkpoint_dir)
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    print(
        "[train] start | "
        f"env={env_id} ({env_label}) | "
        f"action_mode={action_table_mode} ({action_size} actions) | "
        f"device={device} | "
        f"lr={LR} gamma={GAMMA} | "
        f"batch={BATCH_SIZE} warm_up={WARM_UP_STEPS} target_update={TARGET_UPDATE_STEPS} | "
        f"eps_decay_factor={epsilon_decay_factor} memory={memory_size} | "
        f"episodes={episodes} | "
        f"planner_warm_up={use_planner_warm_up} "
        f"warm_up_episodes={planner_warm_up_episodes} "
        f"grid={planner_grid_size} | "
        f"train_every_steps={train_every_steps} | "
        f"alternating_warmup={enable_alternating_warmup} | "
        f"warmup_test_episodes={warmup_test_episodes}"
    )

    rewards: List[float] = []
    epsilon_per_episode: List[float] = []
    planner_warmup_rewards: List[float] = []
    agent_eval_warmup_rewards: List[float] = []
    warmup_test_rewards: List[float] = []

    train_start = time.perf_counter()
    warm_up_episodes = int(planner_warm_up_episodes) if use_planner_warm_up else 0
    total_episodes = warm_up_episodes + int(episodes)
    for ep in range(total_episodes):
        in_planner_warm_up = ep < warm_up_episodes
        use_alternation = in_planner_warm_up and bool(enable_alternating_warmup)
        planner_turn = True
        if use_alternation:
            planner_turn = ep % 2 == 0
        run_planner_episode = in_planner_warm_up and planner_turn
        run_agent_eval_episode = in_planner_warm_up and not planner_turn
        if in_planner_warm_up:
            agent.epsilon = 1.0
        else:
            if ep == warm_up_episodes:
                agent.episodes_since_decay_start = 0
                print(
                    f"[train] planner warm-up complete after {warm_up_episodes} episodes | "
                    "switching to standard epsilon-decay DQN"
                )
                if checkpoint_root is not None and warm_up_episodes > 0:
                    warm_up_path = checkpoint_path(
                        env_label=env_label,
                        checkpoint_type="planner_warmup",
                        action_table_mode=action_table_mode,
                        checkpoint_dir=checkpoint_root,
                    )
                    warmup_saved = save_agent_checkpoint(
                        agent,
                        warm_up_path,
                        checkpoint_type="planner_warmup",
                        env_id=env_id,
                        env_label=env_label,
                        action_table_mode=action_table_mode,
                        train_episodes_completed=warm_up_episodes,
                        training_rewards=rewards,
                        training_epsilon=epsilon_per_episode,
                        planner_warmup_rewards=planner_warmup_rewards,
                        agent_eval_warmup_rewards=agent_eval_warmup_rewards,
                        warmup_test_rewards=warmup_test_rewards,
                    )
                    print(f"[checkpoint] saved warm-up -> {warmup_saved}")
                if warmup_test_episodes > 0:
                    print(
                        f"[warm-up-test] start | episodes={warmup_test_episodes} | "
                        "evaluation right after warm-up"
                    )
                    warmup_test_rewards = test_dqn(
                        agent,
                        episodes=warmup_test_episodes,
                        rng_seed=rng_seed,
                        env_seed=env_base_seed,
                        env_id=env_id,
                        env_label=f"{env_label}_warmup",
                        action_table_mode=action_table_mode,
                        max_episode_steps=max_episode_steps,
                    )
            agent.update_epsilon(agent.episodes_since_decay_start)
        episode_seed = env_base_seed if ep == 0 else None
        state, _ = env.reset(seed=episode_seed)
        done = False
        total_reward = 0.0
        if run_agent_eval_episode:
            agent.epsilon = 0.0

        while not done:
            if run_planner_episode:
                action = agent.select_planned_action(state)
            else:
                action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            if not run_agent_eval_episode:
                agent.store((state, action, float(reward), next_state, done))
            agent.steps_done += 1
            if not run_agent_eval_episode and agent.steps_done % train_every_steps == 0:
                agent.train_step()

            state = next_state
            total_reward += float(reward)

        rewards.append(total_reward)
        epsilon_per_episode.append(agent.epsilon)
        if run_planner_episode:
            planner_warmup_rewards.append(total_reward)
        elif run_agent_eval_episode:
            agent_eval_warmup_rewards.append(total_reward)
        if not in_planner_warm_up:
            agent.episodes_since_decay_start += 1

        if ep % 50 == 0 or ep == total_episodes - 1:
            tail = rewards[-50:]
            tail_mean = float(np.mean(tail))
            tail_success = 100.0 * _success_rate(tail)
            if run_planner_episode:
                phase = "warm-up/planner-train"
            elif run_agent_eval_episode:
                phase = "warm-up/agent-eval"
            else:
                phase = "train"
            print(
                f"[{phase}] ep {ep}/{total_episodes - 1} | "
                f"last_reward={total_reward:.0f} | "
                f"mean{len(tail)}={tail_mean:.3f} | "
                f"success_last{len(tail)}={tail_success:.1f}% | "
                f"epsilon={agent.epsilon:.3f} | "
                f"steps={agent.steps_done} | "
                f"learn={agent.learn_steps}"
            )

    env.close()
    agent.training_episodes = total_episodes
    agent.training_time_seconds = time.perf_counter() - train_start
    agent.training_rewards = rewards
    agent.training_epsilon = epsilon_per_episode
    agent.planner_warmup_rewards = planner_warmup_rewards
    agent.agent_eval_warmup_rewards = agent_eval_warmup_rewards
    agent.warmup_test_rewards = warmup_test_rewards

    success_rate = 100.0 * _success_rate(rewards)
    print(
        "[train] summary | "
        f"episodes={total_episodes} | "
        f"success_rate_all={success_rate:.1f}% | "
        f"mean_reward={float(np.mean(rewards)):.3f} | "
        f"time={agent.training_time_seconds:.2f}s | "
        f"planner_calls={agent.planner_calls} "
        f"planner_cache_hits={agent.planner_cache_hits} "
        f"planner_misses={agent.planner_misses} "
        f"planner_time={agent.planner_time_seconds:.2f}s | "
        f"planner_warmup_eps={len(planner_warmup_rewards)} "
        f"agent_eval_warmup_eps={len(agent_eval_warmup_rewards)}"
    )

    if checkpoint_root is not None:
        final_path = checkpoint_path(
            env_label=env_label,
            checkpoint_type="trained",
            action_table_mode=action_table_mode,
            checkpoint_dir=checkpoint_root,
        )
        saved = save_agent_checkpoint(
            agent,
            final_path,
            checkpoint_type="trained",
            env_id=env_id,
            env_label=env_label,
            action_table_mode=action_table_mode,
            train_episodes_completed=total_episodes,
            training_rewards=rewards,
            training_epsilon=epsilon_per_episode,
            planner_warmup_rewards=planner_warmup_rewards,
            agent_eval_warmup_rewards=agent_eval_warmup_rewards,
            warmup_test_rewards=warmup_test_rewards,
        )
        print(f"[checkpoint] saved -> {saved}")

    return agent


def test_dqn(
    agent: DQNAgent,
    episodes: int,
    rng_seed: int,
    env_seed: Optional[int],
    env_id: str = ENV_ID,
    env_label: str = "Medium_sparse",
    action_table_mode: str = "nine",
    max_episode_steps: Optional[int] = None,
) -> List[float]:
    set_seed(rng_seed)
    env_base_seed = env_seed

    env = _make_env(
        env_id=env_id,
        action_table_mode=action_table_mode,
        max_episode_steps=max_episode_steps,
    )
    eval_epsilon = agent.epsilon
    agent.epsilon = 0.0

    episode_rewards: List[float] = []
    for ep in range(episodes):
        state, _ = env.reset(seed=_evaluation_episode_seed(env_base_seed, ep))
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
        episode_rewards.append(total_reward)

    env.close()
    agent.epsilon = eval_epsilon

    successes = sum(1 for r in episode_rewards if r > 0.0)
    mean_reward = float(np.mean(episode_rewards))
    last_n = min(50, episodes)
    success_last_n = 100.0 * _success_rate(episode_rewards[-last_n:])
    print(
        "[test] summary | "
        f"env={env_label} | "
        f"episodes={episodes} | "
        f"success_rate_all={100.0 * successes / episodes:.1f}% | "
        f"success_rate_last{last_n}={success_last_n:.1f}% | "
        f"mean_reward={mean_reward:.3f}"
    )
    return episode_rewards

