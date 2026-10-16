"""HER terminal correction for sparse, episodic PointMaze-v3 only.

SB3 2.7.1 relabels goals/rewards but keeps original dones. For this task,
success is both reward=1 and a true terminal state. A relabelled transition
must therefore end exactly when it reaches its NEW goal, not the old one.
Timeouts (or an old-goal terminal that isn't terminal for the new goal) still
permit bootstrapping. The original replay arrays are never changed.
"""
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


class EpisodicPointMazeHER(HerReplayBuffer):
    def _get_virtual_samples(self, batch_indices, env_indices, env=None):
        # Reward normalization would invalidate reward==success. Explicitly
        # reject it rather than silently corrupting terminal masks.
        if env is not None:
            raise ValueError("EpisodicPointMazeHER does not support VecNormalize")
        samples = super()._get_virtual_samples(batch_indices, env_indices, env)
        return samples._replace(dones=(samples.rewards > 0).float())
