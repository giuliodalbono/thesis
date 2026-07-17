import random
import time
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from slippery_probability_wrapper import SlipperyProbabilityWrapper

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
except ImportError as exc:
    raise ImportError(
        "stable-baselines3 is required for SAC experiments. "
        "Install it with: pip install stable-baselines3"
    ) from exc


ENV_ID = "FrozenLake-v1"
P_NOMINAL = 0.8
P_SLIP = 0.1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluation_episode_seed(base_seed, episode_idx):
    return None if base_seed is None else base_seed + episode_idx


class OneHotObservationWrapper(gym.ObservationWrapper):
    """Convert Discrete observations into one-hot float vectors."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Discrete):
            raise TypeError("OneHotObservationWrapper requires a Discrete observation space.")
        self._n_obs = int(env.observation_space.n)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._n_obs,),
            dtype=np.float32,
        )

    def observation(self, observation):
        vec = np.zeros(self._n_obs, dtype=np.float32)
        vec[int(observation)] = 1.0
        return vec


class ContinuousToDiscreteActionWrapper(gym.ActionWrapper):
    """
    Expose a continuous 1-D action space and quantize it to discrete actions.
    This lets SAC (continuous-control algorithm) interact with FrozenLake.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Discrete):
            raise TypeError("ContinuousToDiscreteActionWrapper requires a Discrete action space.")
        self._n_actions = int(env.action_space.n)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

    def action(self, action):
        value = float(np.asarray(action).reshape(-1)[0])
        clipped = np.clip(value, -1.0, 1.0)
        scaled = (clipped + 1.0) / 2.0  # [0, 1]
        idx = int(np.floor(scaled * self._n_actions))
        if idx >= self._n_actions:
            idx = self._n_actions - 1
        return idx


def _make_env(map_desc):
    base_env = gym.make(ENV_ID, is_slippery=True, desc=map_desc)
    wrapped = SlipperyProbabilityWrapper(base_env, p_nominal=P_NOMINAL, p_slip=P_SLIP)
    wrapped = ContinuousToDiscreteActionWrapper(wrapped)
    wrapped = OneHotObservationWrapper(wrapped)
    return wrapped


def _make_vec_env(map_desc, env_seed):
    def _init():
        env = _make_env(map_desc)
        if env_seed is not None:
            env.reset(seed=env_seed)
        return env

    vec_env = DummyVecEnv([_init])
    return VecMonitor(vec_env)


class EpisodeRewardCallback(BaseCallback):
    """Collect per-episode rewards from vec env info dicts."""

    def __init__(self):
        super().__init__()
        self.episode_rewards: List[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if episode is not None:
                self.episode_rewards.append(float(episode["r"]))
        return True


def train_sac(
    total_timesteps: int,
    rng_seed: int,
    env_seed,
    map_desc,
    map_name: str,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    train_freq: int = 1,
    gradient_steps: int = 1,
    ent_coef="auto",
    device: str = "auto",
) -> Tuple[SAC, Dict[str, float]]:
    set_seed(rng_seed)
    env = _make_vec_env(map_desc, env_seed)
    callback = EpisodeRewardCallback()

    print(
        "[sac train] start | "
        f"map={map_name} | "
        f"total_timesteps={total_timesteps} | "
        f"lr={learning_rate} gamma={gamma} batch_size={batch_size} "
        f"tau={tau} train_freq={train_freq} gradient_steps={gradient_steps} "
        f"ent_coef={ent_coef}"
    )

    model = SAC(
        "MlpPolicy",
        env,
        seed=rng_seed,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        ent_coef=ent_coef,
        verbose=0,
        device=device,
    )

    start = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    elapsed = time.perf_counter() - start

    metrics = {
        "training_time_seconds": elapsed,
        "episodes": len(callback.episode_rewards),
    }

    print(
        "[sac train] summary | "
        f"map={map_name} | "
        f"episodes={metrics['episodes']} | "
        f"time={elapsed:.2f}s"
    )

    return model, {
        "training_rewards": callback.episode_rewards,
        "training_time_seconds": elapsed,
        "episodes": metrics["episodes"],
    }


def test_sac(
    model: SAC,
    episodes: int,
    rng_seed: int,
    env_seed,
    map_desc,
    map_name: str,
    deterministic: bool = True,
) -> List[float]:
    set_seed(rng_seed)
    env = _make_env(map_desc)

    episode_rewards = []
    for ep in range(episodes):
        state, _ = env.reset(seed=_evaluation_episode_seed(env_seed, ep))
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(state, deterministic=deterministic)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        episode_rewards.append(total_reward)

    env.close()

    wins = sum(1 for r in episode_rewards if r > 0)
    mean_r = sum(episode_rewards) / len(episode_rewards)
    last_n = min(50, episodes)
    win_last_n = 100.0 * (
        sum(1 for r in episode_rewards[-last_n:] if r > 0) / last_n
    )

    print(
        "[sac test] summary | "
        f"map={map_name} | "
        f"episodes={episodes} | "
        f"success_rate_all={100 * wins / episodes:.1f}% | "
        f"success_rate_last{last_n}={win_last_n:.1f}% | "
        f"mean_reward={mean_r:.3f}"
    )

    return episode_rewards
