# ===============================
# Deep Q-Network (DQN) - FrozenLake-v1
# ===============================


import random
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import planner
from slippery_probability_wrapper import SlipperyProbabilityWrapper


def format_frozen_lake_state(env, state: int) -> str:
    """Return an ASCII map with A=agent, S/F/H/G as tile types."""
    desc = env.unwrapped.desc
    ncols = int(env.unwrapped.ncol)
    rows = []

    for r, row in enumerate(desc):
        rendered_row = []
        for c, cell in enumerate(row):
            idx = r * ncols + c
            symbol = cell.decode("utf-8") if isinstance(cell, bytes) else str(cell)
            rendered_row.append("A" if idx == state else symbol)
        rows.append(" ".join(rendered_row))

    return "\n".join(rows)


def pick_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


ENV_ID = "FrozenLake-v1"
GAMMA = 0.99
LR = 1e-3
EPSILON_START_WARM_UP = 1.0
EPSILON_2_START_WARM_UP = 1.0
EPSILON_START_PLANNER = 0.5
EPSILON_2_START_PLANNER = 0
EPSILON_START_NO_PLANNER = 1.0
EPSILON_MIN = 0.05
EPSILON_2_MIN = 0
BATCH_SIZE = 128
WARM_UP_STEPS = 512
TARGET_UPDATE_STEPS = 100
CHECKPOINT_TYPES = {"planner_warmup", "planner_trained", "plain_trained"}


def _validate_decay_factor(factor: float, name: str) -> None:
    if not 0.0 < factor < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {factor}")


def get_checkpoint_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_dir = (
        project_root / "resources" / "planned-deterministic-frozen-lake" / "checkpoints"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def checkpoint_path(map_name: str, checkpoint_type: str, checkpoint_dir=None) -> Path:
    if checkpoint_type not in CHECKPOINT_TYPES:
        raise ValueError(
            f"Unknown checkpoint_type={checkpoint_type!r}. "
            f"Expected one of {sorted(CHECKPOINT_TYPES)}"
        )
    output_dir = get_checkpoint_dir() if checkpoint_dir is None else Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_map_name = str(map_name).replace(" ", "_")
    return output_dir / f"FrozenLake_{safe_map_name}_{checkpoint_type}.pt"


def _build_checkpoint_payload(
    agent,
    *,
    checkpoint_type,
    map_name,
    map_desc,
    planner_warm_up_episodes,
    train_episodes_completed,
    training_rewards,
    training_epsilon,
    training_epsilon_2,
    include_memory,
) -> Dict[str, Any]:
    # TODO: when we migrate to whole-agent persistence (torch.save(agent)),
    # replace this payload builder and simplify load/save paths accordingly.
    return {
        "version": 1,
        "checkpoint_type": checkpoint_type,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "map_name": map_name,
        "map_desc": list(map_desc),
        "planner_warm_up_episodes": int(planner_warm_up_episodes),
        "train_episodes_completed": int(train_episodes_completed),
        "agent_state": {
            "use_planner": bool(agent.use_planner),
            "state_size": int(agent.state_size),
            "action_size": int(agent.action_size),
            "memory_size": int(agent.memory_size),
            "epsilon": float(agent.epsilon),
            "epsilon_2": float(agent.epsilon_2),
            "epsilon_start": float(agent.epsilon_start),
            "epsilon_2_start": float(agent.epsilon_2_start),
            "epsilon_min": float(agent.epsilon_min),
            "epsilon_decay_factor": float(agent.epsilon_decay_factor),
            "epsilon_2_min": float(agent.epsilon_2_min),
            "epsilon_2_decay_factor": float(agent.epsilon_2_decay_factor),
            "steps_done": int(agent.steps_done),
            "learn_steps": int(agent.learn_steps),
            "episodes_since_decay_start": int(agent.episodes_since_decay_start),
            "plan_time_seconds": float(agent.plan_time_seconds),
            "training_time_seconds": float(agent.training_time_seconds),
            "training_episodes": int(agent.training_episodes),
            "plan_time_pct": float(agent.plan_time_pct),
        },
        "training_metrics": {
            "training_rewards": list(training_rewards),
            "training_epsilon": list(training_epsilon),
            "training_epsilon_2": list(training_epsilon_2),
        },
        "model_state_dict": agent.model.state_dict(),
        "target_model_state_dict": agent.target_model.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "memory": list(agent.memory) if include_memory else None,
    }


def save_agent_checkpoint(
    agent,
    path,
    *,
    checkpoint_type,
    map_name,
    map_desc,
    planner_warm_up_episodes,
    train_episodes_completed,
    training_rewards,
    training_epsilon,
    training_epsilon_2,
    include_memory=False,
) -> Path:
    if checkpoint_type not in CHECKPOINT_TYPES:
        raise ValueError(
            f"Unknown checkpoint_type={checkpoint_type!r}. "
            f"Expected one of {sorted(CHECKPOINT_TYPES)}"
        )

    checkpoint_file = Path(path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_checkpoint_payload(
        agent,
        checkpoint_type=checkpoint_type,
        map_name=map_name,
        map_desc=map_desc,
        planner_warm_up_episodes=planner_warm_up_episodes,
        train_episodes_completed=train_episodes_completed,
        training_rewards=training_rewards,
        training_epsilon=training_epsilon,
        training_epsilon_2=training_epsilon_2,
        include_memory=include_memory,
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
        use_planner=agent_state["use_planner"],
        epsilon_decay_factor=agent_state["epsilon_decay_factor"],
        epsilon_2_decay_factor=agent_state["epsilon_2_decay_factor"],
        memory_size=agent_state["memory_size"],
        device=pick_device(),
    )

    agent.model.load_state_dict(payload["model_state_dict"])
    agent.target_model.load_state_dict(payload["target_model_state_dict"])
    agent.optimizer.load_state_dict(payload["optimizer_state_dict"])

    agent.epsilon = float(agent_state["epsilon"])
    agent.epsilon_2 = float(agent_state["epsilon_2"])
    agent.epsilon_start = float(agent_state["epsilon_start"])
    agent.epsilon_2_start = float(agent_state["epsilon_2_start"])
    agent.epsilon_min = float(agent_state["epsilon_min"])
    agent.epsilon_2_min = float(agent_state["epsilon_2_min"])
    agent.steps_done = int(agent_state["steps_done"])
    agent.learn_steps = int(agent_state["learn_steps"])
    agent.episodes_since_decay_start = int(agent_state["episodes_since_decay_start"])
    agent.plan_time_seconds = float(agent_state.get("plan_time_seconds", 0.0))
    agent.training_time_seconds = float(agent_state.get("training_time_seconds", 0.0))
    agent.training_episodes = int(agent_state.get("training_episodes", 0))
    agent.plan_time_pct = float(agent_state.get("plan_time_pct", 0.0))

    memory = payload.get("memory")
    if memory is not None:
        agent.memory = deque(memory, maxlen=agent.memory_size)

    metrics = payload.get("training_metrics", {})
    agent.training_rewards = list(metrics.get("training_rewards", []))
    agent.training_epsilon = list(metrics.get("training_epsilon", []))
    agent.training_epsilon_2 = list(metrics.get("training_epsilon_2", []))

    metadata = {
        "checkpoint_path": str(checkpoint_file),
        "checkpoint_type": payload.get("checkpoint_type"),
        "map_name": payload.get("map_name"),
        "map_desc": payload.get("map_desc"),
        "planner_warm_up_episodes": payload.get("planner_warm_up_episodes", 0),
        "train_episodes_completed": payload.get("train_episodes_completed", 0),
        "saved_at": payload.get("saved_at"),
    }
    return agent, metadata


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
    def __init__(
        self,
        state_size,
        action_size,
        use_planner,
        epsilon_decay_factor,
        epsilon_2_decay_factor,
        memory_size,
        device=None
    ):
        _validate_decay_factor(epsilon_decay_factor, "epsilon_decay_factor")
        _validate_decay_factor(epsilon_2_decay_factor, "epsilon_2_decay_factor")

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = GAMMA
        if use_planner:
            self.epsilon_start = EPSILON_START_PLANNER
            self.epsilon_2_start = EPSILON_2_START_PLANNER
            self.epsilon = EPSILON_START_PLANNER
            self.epsilon_2 = EPSILON_2_START_PLANNER
        else:
            self.epsilon_start = EPSILON_START_NO_PLANNER
            self.epsilon_2_start = EPSILON_2_START_PLANNER
            self.epsilon = EPSILON_START_NO_PLANNER
            self.epsilon_2 = EPSILON_2_START_PLANNER
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay_factor = epsilon_decay_factor
        self.epsilon_2_min = EPSILON_2_MIN
        self.epsilon_2_decay_factor = epsilon_2_decay_factor
        self.batch_size = BATCH_SIZE
        self.warm_up_steps = WARM_UP_STEPS
        self.target_update_steps = TARGET_UPDATE_STEPS
        self.memory_size = memory_size
        self.episodes_since_decay_start = 0

        self.use_planner = use_planner
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
        self.planner_calls = 0
        self.planner_misses = 0
        self.planner_cache_hits = 0
        self._plan_cache: Dict[int, Optional[List[str]]] = {}
        self.plan_time_seconds = 0.0
        self.training_time_seconds = 0.0
        self.training_episodes = 0
        self.plan_time_pct = 0.0

    def one_hot(self, state):
        vec = np.zeros(self.state_size, dtype=np.float32)
        vec[state] = 1.0
        return vec

    def select_action(self, env, state):
        if random.random() < self.epsilon:
            return self.select_explore_action(env, state)

        return self.select_exploit_action(state)

    def select_explore_action(self, env, state):
        if self.use_planner:
            if random.random() < self.epsilon_2:
                return self.select_planned_action(env, state)

        return random.randrange(self.action_size)

    def select_exploit_action(self, state):
        state_t = torch.tensor(self.one_hot(state)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def select_planned_action(self, env, state):
        planned_action = self.get_planned_action(env, state)
        if planned_action is None:
            return random.randrange(self.action_size)
        return planned_action

    def get_planned_action(self, env, state):
        t0 = time.perf_counter()
        try:
            if state in self._plan_cache:
                self.planner_cache_hits += 1
                plan = self._plan_cache[state]
            else:
                self.planner_calls += 1
                problem = planner.define_problem(env, state)
                plan = planner.build_plan(problem)
                self._plan_cache[state] = plan
                if plan is None or len(plan) == 0:
                    self.planner_misses += 1

            if plan is None or len(plan) == 0:
                return None

            # "move_{s}_{next_state}_{a}" -> gym_action
            parts = plan[0].split("_")
            return int(parts[-1])
        finally:
            self.plan_time_seconds += time.perf_counter() - t0

    def store(self, transition):
        self.memory.append(transition)

    def update_epsilon(self):
        ep = self.episodes_since_decay_start
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_start * (self.epsilon_decay_factor ** ep),
        )
        self.epsilon_2 = max(
            self.epsilon_2_min,
            self.epsilon_2_start * (self.epsilon_2_decay_factor ** ep),
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
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


def _evaluation_episode_seed(base_seed, episode_idx):
    return None if base_seed is None else base_seed + episode_idx


def _make_env(map_desc):
    base_env = gym.make(ENV_ID, is_slippery=True, desc=map_desc)
    return SlipperyProbabilityWrapper(base_env, p_nominal=0.8, p_slip=0.1)


def train_dqn(
    episodes,
    use_planner,
    rng_seed,
    env_seed,
    map_desc,
    map_name,
    epsilon_decay_factor,
    epsilon_2_decay_factor,
    memory_size,
    planner_warm_up_episodes=0,
    log_every=50,
    checkpoint_dir=None,
    initial_agent=None,
    on_planner_warmup_complete: Optional[Callable[..., None]] = None,
):
    set_seed(rng_seed)
    env_base = env_seed

    env = _make_env(map_desc)

    state_size = env.observation_space.n
    action_size = env.action_space.n

    if initial_agent is not None:
        if planner_warm_up_episodes != 0:
            raise ValueError(
                "planner_warm_up_episodes must be 0 when using initial_agent."
            )
        if initial_agent.state_size != state_size or initial_agent.action_size != action_size:
            raise ValueError("initial_agent is incompatible with the selected map.")
        if initial_agent.use_planner != use_planner:
            raise ValueError("initial_agent use_planner does not match train_dqn use_planner.")
        agent = initial_agent
        device = agent.device
    else:
        device = pick_device()
        agent = DQNAgent(
            state_size,
            action_size,
            use_planner=use_planner,
            epsilon_decay_factor=epsilon_decay_factor,
            epsilon_2_decay_factor=epsilon_2_decay_factor,
            memory_size=memory_size,
            device=device,
        )

    warm_up_episodes = planner_warm_up_episodes if use_planner else 0
    total_episodes = episodes + warm_up_episodes
    checkpoint_root = (
        None
        if checkpoint_dir is None
        else Path(checkpoint_dir)
    )
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    print(
        "[train] start | "
        f"map={map_name} | "
        f"device={device} | "
        f"lr={LR} gamma={GAMMA} | "
        f"batch={BATCH_SIZE} warm_up={WARM_UP_STEPS} target_update={TARGET_UPDATE_STEPS} | "
        f"eps_decay_factor={epsilon_decay_factor} eps2_decay_factor={epsilon_2_decay_factor} "
        f"memory={memory_size} | "
        f"episodes={episodes}"
        + (
            f" | planner_warm_up_episodes={warm_up_episodes}"
            if warm_up_episodes
            else ""
        )
    )

    rewards = []
    epsilon_per_episode: List[float] = []
    epsilon_2_per_episode: List[float] = []

    train_start = time.perf_counter()
    for ep in range(total_episodes):
        in_planner_warm_up = ep < warm_up_episodes

        if in_planner_warm_up:
            agent.epsilon = EPSILON_START_WARM_UP
            agent.epsilon_2 = EPSILON_2_START_WARM_UP
        elif ep == warm_up_episodes and warm_up_episodes > 0:
            agent.episodes_since_decay_start = 0
            agent.epsilon_start = EPSILON_START_PLANNER
            agent.epsilon_2_start = EPSILON_2_START_PLANNER
            agent.update_epsilon()
            print(
                f"[train] planner warm-up complete after {warm_up_episodes} episodes | "
                f"steps={agent.steps_done} | starting normal training"
            )
            warmup_checkpoint_file = None
            if checkpoint_root is not None and use_planner:
                warmup_path = checkpoint_path(
                    map_name=map_name,
                    checkpoint_type="planner_warmup",
                    checkpoint_dir=checkpoint_root,
                )
                saved = save_agent_checkpoint(
                    agent,
                    warmup_path,
                    checkpoint_type="planner_warmup",
                    map_name=map_name,
                    map_desc=map_desc,
                    planner_warm_up_episodes=warm_up_episodes,
                    train_episodes_completed=len(rewards),
                    training_rewards=rewards,
                    training_epsilon=epsilon_per_episode,
                    training_epsilon_2=epsilon_2_per_episode,
                    include_memory=True,
                )
                warmup_checkpoint_file = saved
                print(f"[checkpoint] saved -> {saved}")
            if use_planner and on_planner_warmup_complete is not None:
                on_planner_warmup_complete(
                    checkpoint_path=warmup_checkpoint_file,
                    map_name=map_name,
                    map_desc=map_desc,
                    rng_seed=rng_seed,
                    env_seed=env_seed,
                )
        elif not in_planner_warm_up:
            agent.update_epsilon()

        # Seed only the first reset so the run is reproducible without replaying
        # the same slippery outcomes at the start of every episode.
        episode_seed = env_base if ep == 0 else None
        state, _ = env.reset(seed=episode_seed)
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(env, state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store((state, action, reward, next_state, done))

            agent.steps_done += 1
            agent.train_step()

            state = next_state
            total_reward += reward

        if not in_planner_warm_up:
            agent.episodes_since_decay_start += 1

        rewards.append(total_reward)
        epsilon_per_episode.append(agent.epsilon)
        epsilon_2_per_episode.append(agent.epsilon_2)

        if ep % log_every == 0 or ep == total_episodes - 1:
            tail = rewards[-50:]
            win_tail = 100 * _win_rate(tail)
            phase = "warm-up" if in_planner_warm_up else "train"
            print(
                f"[{phase}] "
                f"ep {ep}/{total_episodes - 1} | "
                f"last_reward={total_reward:.0f} | "
                f"mean{len(tail)}={sum(tail) / len(tail):.3f} | "
                f"win_last{len(tail)}={win_tail:.0f}% | "
                f"epsilon={agent.epsilon:.3f} | "
                f"epsilon_2={agent.epsilon_2:.3f} | "
                f"steps={agent.steps_done} | "
                f"learn={agent.learn_steps}"
            )

    wins = sum(1 for r in rewards if r > 0)
    env.close()

    agent.training_episodes = total_episodes
    agent.training_time_seconds = time.perf_counter() - train_start
    if agent.training_time_seconds > 0:
        agent.plan_time_pct = (
            100.0 * agent.plan_time_seconds / agent.training_time_seconds
        )
    else:
        agent.plan_time_pct = 0.0

    last_n = min(50, total_episodes)
    tail_final = rewards[-last_n:]
    win_last_n = 100 * _win_rate(tail_final)

    parts = [
        "[train] summary | "
        f"win_rate_all={100 * wins / total_episodes:.1f}% | "
        f"win_rate_last{last_n}={win_last_n:.1f}% | "
        f"mean_reward={sum(rewards) / len(rewards):.3f} | "
        f"time={agent.training_time_seconds:.2f}s",
    ]
    if agent.use_planner:
        pm = agent.planner_misses
        pc = agent.planner_calls
        miss_pct = 100.0 * pm / pc if pc else 0.0
        parts.append(
            f"plan_time={agent.plan_time_seconds:.2f}s "
            f"({agent.plan_time_pct:.2f}% of training) | "
            f"planner_solves={pc} cache_hits={agent.planner_cache_hits} "
            f"miss={pm}/{pc} ({miss_pct:.1f}%)"
        )
    print(" | ".join(parts))

    agent.training_rewards = rewards
    agent.training_epsilon = epsilon_per_episode
    agent.training_epsilon_2 = epsilon_2_per_episode

    if checkpoint_root is not None:
        final_type = "planner_trained" if use_planner else "plain_trained"
        final_path = checkpoint_path(
            map_name=map_name,
            checkpoint_type=final_type,
            checkpoint_dir=checkpoint_root,
        )
        saved = save_agent_checkpoint(
            agent,
            final_path,
            checkpoint_type=final_type,
            map_name=map_name,
            map_desc=map_desc,
            planner_warm_up_episodes=warm_up_episodes,
            train_episodes_completed=total_episodes,
            training_rewards=rewards,
            training_epsilon=epsilon_per_episode,
            training_epsilon_2=epsilon_2_per_episode,
            include_memory=False,
        )
        print(f"[checkpoint] saved -> {saved}")
    return agent


# ===============================
# TEST
# ===============================
def test_dqn(agent, episodes, rng_seed, env_seed, map_desc, map_name):
    set_seed(rng_seed)
    env_base = env_seed

    env = _make_env(map_desc)
    agent.epsilon = 0.0
    agent.epsilon_2 = 0.0
    agent.steps_done = 0
    agent.planner_calls = 0
    agent.planner_misses = 0
    agent.planner_cache_hits = 0

    episode_rewards = []
    for ep in range(episodes):
        # Reuse the same deterministic seed schedule across agents so evaluation
        # stays fair while still covering different stochastic episodes.
        state, _ = env.reset(seed=_evaluation_episode_seed(env_base, ep))
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
        f"map={map_name} | "
        f"episodes={episodes} | "
        f"success_rate_all={100 * wins / episodes:.1f}% | "
        f"success_rate_last{last_n}={win_last_n:.1f}% | "
        f"mean_reward={mean_r:.3f} | "
        f"steps_done={agent.steps_done}"
    )
    return episode_rewards
