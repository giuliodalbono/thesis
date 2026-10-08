"""Temporary evaluation state for pauses in a cumulative training run."""
from contextlib import contextmanager
from copy import deepcopy
import random
import numpy as np
import torch


@contextmanager
def preserve_training_state(agent):
    """Restore random generators and recovery state after a read-only test.

    Evaluation must still use learn=False: this guard does not copy expensive
    replay/model/optimizer data. Tests assert that those remain unchanged.
    Recovery can take random actions even at epsilon=0, hence RNG restoration
    is necessary for the resumed training trajectory to be reproducible.
    """
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    agent_rng = agent.rng.getstate()
    recovery = {key: deepcopy(value) for key, value in vars(agent).items()
                if key.startswith("_") or key in ("recovery_events", "recovery_actions")}
    modes = agent.online.training, agent.target.training
    try:
        agent.online.eval()
        agent.target.eval()
        yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        agent.rng.setstate(agent_rng)
        for key, value in recovery.items():
            setattr(agent, key, value)
        agent.online.train(modes[0])
        agent.target.train(modes[1])
