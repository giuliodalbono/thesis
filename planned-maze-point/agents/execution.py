"""Scoped execution settings: a SAC run must not change later DQN runs."""
from contextlib import contextmanager
import torch


@contextmanager
def agent_execution(config):
    previous = torch.get_num_threads()
    try:
        if config.agent_name == "sac_her":
            # Small MLPs often spend more time on thread coordination than math.
            torch.set_num_threads(config.torch_threads)
        yield
    finally:
        if torch.get_num_threads() != previous:
            torch.set_num_threads(previous)
