"""Planned versus plain DQN orchestration with independently sized training phases.

Both agents start from identically seeded weights. Only planned receives planner
transitions first. Each normal-training epsilon schedule starts at its own episode
zero; final tests use matched environment seeds. Phase order and learning rules
are preserved from the previous service.
"""
from __future__ import annotations
from typing import Dict, List, Optional
from configurations import PointMazeConfig, PointMazeOpenConfig
from agents import create_agent
from agents.exploration import training_epsilon, epsilon_decay_horizon
from environments.common import flatten_observation
from planners import create_planner
from .runner import run_episode, summarize


def run_planned_vs_plain(config, **budgets):
    """Algorithm-neutral entry point; keep the legacy DQN service callable."""
    if config.agent_name == "sac_her":
        from .sac_comparison import run_planned_vs_plain_sac
        return run_planned_vs_plain_sac(config, **budgets)
    return run_planned_vs_plain_dqn(config, **budgets)

def _normal_training(
    env,
    agent,
    config: PointMazeConfig,
    episodes: int,
    reset_seed_offset: int,
) -> List[Dict[str, object]]:
    return [
        run_episode(
            env,
            agent,
            config.seed + reset_seed_offset + index,
            planner=None,
            learn=True,
            config=config,
            epsilon=training_epsilon(config, index, episodes),
        )
        for index in range(episodes)
    ]


def _evaluation(
    env,
    agent,
    config: PointMazeConfig,
    episodes: int,
    reset_seed_offset: int,
) -> List[Dict[str, object]]:
    return [
        run_episode(
            env,
            agent,
            config.seed + reset_seed_offset + index,
            planner=None,
            learn=False,
            config=config,
        )
        for index in range(episodes)
    ]


def run_planned_vs_plain_dqn(
    config: Optional[PointMazeConfig] = None,
    planner_warmup_episodes: int = 0,
    planned_train_episodes: int = 0,
    plain_train_episodes: int = 0,
    test_episodes: int = 500,
) -> Dict[str, object]:
    """Compare DQN with planner-only warm-up against an otherwise identical DQN.

    Planned DQN: planner-only warm-up, a greedy post-warm-up test when normal
    training is requested, then epsilon-greedy normal DQN training and a final
    greedy test.
    Plain DQN: no planner warm-up, then its independently sized DQN training.
    Both agents use the same deterministic evaluation tasks with epsilon fixed at zero.
    """
    if min(planner_warmup_episodes, planned_train_episodes, plain_train_episodes, test_episodes) < 0:
        raise ValueError("Phase episode counts cannot be negative")
    if test_episodes == 0:
        raise ValueError("test_episodes must be positive")

    config = config or PointMazeOpenConfig()
    if config.agent_name != "dqn":
        raise ValueError("Use run_planned_vs_plain for non-DQN agents")
    config.validate()
    env = config.environment.create(config.max_episode_steps)
    try:
        observation, _ = env.reset(seed=config.seed)
        state_size, action_size = flatten_observation(observation).size, env.action_space.n

        planned_agent = create_agent(config, state_size, action_size)
        planner = create_planner(config)
        planned_warmup = [
            run_episode(env, planned_agent, config.seed + index, planner, True, config)
            for index in range(planner_warmup_episodes)
        ]
        planned_post_warmup_test = None
        if planned_train_episodes > 0:
            planned_post_warmup_test = _evaluation(
                env,
                planned_agent,
                config,
                test_episodes,
                reset_seed_offset=300_000,
            )
        planned_training = _normal_training(env, planned_agent, config, planned_train_episodes, 200_000)
        planned_final_test = _evaluation(
            env,
            planned_agent,
            config,
            test_episodes,
            reset_seed_offset=400_000,
        )

        plain_agent = create_agent(config, state_size, action_size)
        plain_training = _normal_training(env, plain_agent, config, plain_train_episodes, 200_000)
        # This seed set matches the planned agent's final test, not its diagnostic
        # post-warm-up test.
        plain_test = _evaluation(
            env,
            plain_agent,
            config,
            test_episodes,
            reset_seed_offset=400_000,
        )
    finally:
        env.close()

    planned_steps = summarize(planned_warmup + planned_training)["environment_steps"]
    plain_steps = summarize(plain_training)["environment_steps"]
    return {
        "config": config.to_dict(),
        "planner_warmup_episodes": planner_warmup_episodes,
        "planned_train_episodes": planned_train_episodes,
        "plain_train_episodes": plain_train_episodes,
        "epsilon_decay_fraction": config.epsilon_decay_fraction,
        "test_episodes": test_episodes,
        "planned": {
            "warmup": planned_warmup,
            "warmup_summary": summarize(planned_warmup),
            "post_warmup_test": planned_post_warmup_test,
            "post_warmup_test_summary": (
                summarize(planned_post_warmup_test)
                if planned_post_warmup_test is not None
                else None
            ),
            "training": planned_training,
            "training_summary": summarize(planned_training),
            "post_training_test": planned_final_test if planned_train_episodes > 0 else None,
            "post_training_test_summary": (
                summarize(planned_final_test) if planned_train_episodes > 0 else None
            ),
            # Kept as the final planned test for existing callers and for the
            # direct planned-vs-plain comparison.
            "test": planned_final_test,
            "test_summary": summarize(planned_final_test),
            "pre_test_environment_steps": planned_steps,
            "epsilon_decay_horizon_episodes": epsilon_decay_horizon(
                config, planned_train_episodes
            ),
            "optimizer_updates": planned_agent.updates,
        },
        "plain": {
            "training": plain_training,
            "training_summary": summarize(plain_training),
            "test": plain_test,
            "test_summary": summarize(plain_test),
            "pre_test_environment_steps": plain_steps,
            "epsilon_decay_horizon_episodes": epsilon_decay_horizon(
                config, plain_train_episodes
            ),
            "optimizer_updates": plain_agent.updates,
        },
    }
