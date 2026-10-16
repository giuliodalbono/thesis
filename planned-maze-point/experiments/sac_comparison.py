"""Planner-initialized vs plain SAC+HER, with isolated deterministic tests.

Budgets count real episodes (HER adds no environment interactions). Equal
episode budgets are NOT equal step or compute budgets; both are reported.
Validation checkpoints use one matched task set, final tests a held-out set.
"""
import importlib.metadata
import platform
import time
from agents import create_agent
from agents.evaluation import preserve_training_state
from agents.execution import agent_execution
from configurations import PointMazeMediumSACConfig
from planners import create_planner
from .runner import run_episode, summarize


def phase_summary(history, updates=0):
    planner_actions = sum(row.get("planner_actions", 0) for row in history)
    agent_actions = sum(row.get("agent_actions", 0) for row in history)
    return dict(summarize(history),
                elapsed_seconds=sum(row["elapsed_seconds"] for row in history),
                planner_seconds=sum(row["planner_seconds"] for row in history),
                planner_actions=planner_actions, agent_actions=agent_actions,
                planner_action_fraction=(planner_actions / (planner_actions + agent_actions)
                                         if planner_actions + agent_actions else 0.0),
                optimizer_updates=updates)


def warmup_planner_probability(mode, episode_index, total_episodes):
    """Return the per-action planner probability for one warm-up episode.

    ``linear_warmup_anneal`` is a new method condition, not a replacement for
    planner-only warm-up: p=1 at its first episode and p=0 at its last. The
    draw is made for every action, allowing gradual within-episode handover.
    """
    if mode in ("planner_only", "planner_action_noise", "planner_behavior_cloning"):
        return 1.0
    if mode == "linear_warmup_anneal":
        if total_episodes <= 1:
            return 1.0
        return 1.0 - episode_index / (total_episodes - 1)
    raise ValueError(
        "planner_guidance_mode must be 'planner_only', 'linear_warmup_anneal', "
        "or 'planner_action_noise'"
    )


def warmup_planner_noise_std(mode, episode_index, total_episodes, initial_std):
    """Linearly anneal planner-centred continuous exploration to zero."""
    if initial_std < 0.0:
        raise ValueError("planner_action_noise_initial_std must be nonnegative")
    if mode != "planner_action_noise":
        return 0.0
    if total_episodes <= 1:
        return initial_std
    return initial_std * (1.0 - episode_index / (total_episodes - 1))


def behavior_cloning_weight(mode, initial_weight, final_weight, normal_episode, total_episodes):
    """BC is full-strength in planner warm-up and linearly decays afterwards."""
    if initial_weight < 0.0 or final_weight < 0.0:
        raise ValueError("Behaviour-cloning weights must be nonnegative")
    if mode != "planner_behavior_cloning":
        return 0.0
    if total_episodes <= 1:
        return initial_weight
    return initial_weight + (final_weight - initial_weight) * normal_episode / (total_episodes - 1)


def evaluate_sac(env, agent, config, episodes, seed_offset=300_000):
    before = agent.stored_environment_steps, agent.updates
    with preserve_training_state(agent):
        history = [run_episode(env, agent, config.seed + seed_offset + i,
                               None, False, config) for i in range(episodes)]
    if before != (agent.stored_environment_steps, agent.updates):
        raise RuntimeError("Evaluation changed SAC learning counters")
    return history


def run_planned_vs_plain_sac(config=None, planner_warmup_episodes=0,
                             planned_train_episodes=0, plain_train_episodes=0,
                             test_episodes=500, evaluation_interval_episodes=0,
                             planner_guidance_mode="planner_only",
                             planner_action_noise_initial_std=0.0,
                             behavior_cloning_initial_weight=0.0,
                             behavior_cloning_final_weight=0.0,
                             match_plain_interaction_budget=False):
    """One paired seed; warm-up and either training phase may be zero.

    Optional intermediate tests measure policy success, NOT stochastic rollout
    success. They are excluded from training budgets and timed separately.
    A zero-warm-up evaluation is explicitly an untrained-policy baseline.

    When ``match_plain_interaction_budget`` is true, the planned arm runs first
    and the plain arm collects episodes until it reaches the planned arm's
    pre-evaluation environment-step count.  This is more meaningful than
    matching episode counts because successful planner trajectories are shorter.
    """
    config = config or PointMazeMediumSACConfig()
    config.validate()
    if config.agent_name != "sac_her":
        raise ValueError("Use a SAC+HER preset")
    counts = (planner_warmup_episodes, planned_train_episodes, plain_train_episodes,
              test_episodes, evaluation_interval_episodes)
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts) or test_episodes < 1:
        raise ValueError("Episode counts must be nonnegative integers; tests must be positive")
    if not isinstance(match_plain_interaction_budget, bool):
        raise ValueError("match_plain_interaction_budget must be a boolean")
    # Validate even when warm-up is zero, so saved reports never contain an
    # unsupported method label.
    warmup_planner_probability(planner_guidance_mode, 0, planner_warmup_episodes)
    warmup_planner_noise_std(
        planner_guidance_mode, 0, planner_warmup_episodes,
        planner_action_noise_initial_std,
    )
    behavior_cloning_weight(
        planner_guidance_mode, behavior_cloning_initial_weight,
        behavior_cloning_final_weight, 0, planned_train_episodes,
    )
    if planner_guidance_mode == "planner_behavior_cloning" and behavior_cloning_initial_weight <= 0.0:
        raise ValueError("planner_behavior_cloning requires a positive initial BC weight")
    started = time.perf_counter()
    with agent_execution(config):
        env = config.environment.create(config.max_episode_steps, continuous=True)
        evaluation_env = None
        try:
            evaluation_env = config.environment.create(config.max_episode_steps, continuous=True)
            outputs = {}
            for name, warmup_count, train_count in (
                ("planned", planner_warmup_episodes, planned_train_episodes),
                ("plain", 0, plain_train_episodes),
            ):
                # Same seed => identical initial actor/critic weights, but never
                # shared replay, optimizers, or learned parameters across arms.
                uses_bc = name == "planned" and planner_guidance_mode == "planner_behavior_cloning"
                agent = create_agent(config, env=env, behavior_cloning=uses_bc)
                planner = create_planner(config)
                warmup, training, checkpoints = [], [], []
                if uses_bc:
                    agent.set_behavior_cloning_weight(behavior_cloning_initial_weight)
                for i in range(warmup_count):
                    probability = warmup_planner_probability(
                        planner_guidance_mode if name == "planned" else "planner_only",
                        i, warmup_count,
                    )
                    noise_std = warmup_planner_noise_std(
                        planner_guidance_mode if name == "planned" else "planner_only",
                        i, warmup_count, planner_action_noise_initial_std,
                    )
                    warmup.append(run_episode(
                        env, agent, config.seed + i, planner, True, config,
                        planner_probability=probability,
                        planner_action_noise_std=noise_std,
                    ))
                    _progress(config, name + " warm-up", i + 1, warmup_count, agent)
                warmup_updates = agent.updates
                initial_test = evaluate_sac(evaluation_env, agent, config, test_episodes)
                checkpoints.append(_checkpoint(agent, initial_test, 0,
                                                "post_warmup" if warmup_count else "untrained"))
                i = 0
                while (i < train_count if not (name == "plain" and match_plain_interaction_budget)
                       else agent.stored_environment_steps < outputs["planned"]["pre_test_environment_steps"]):
                    if uses_bc:
                        agent.set_behavior_cloning_weight(behavior_cloning_weight(
                            planner_guidance_mode, behavior_cloning_initial_weight,
                            behavior_cloning_final_weight, i, train_count,
                        ))
                    training.append(run_episode(env, agent, config.seed + 200_000 + i,
                                                None, True, config))
                    i += 1
                    progress_total = (outputs["planned"]["pre_test_environment_steps"]
                                      if name == "plain" and match_plain_interaction_budget else train_count)
                    _progress(config, name + " training", i, progress_total, agent,
                              unit="steps" if name == "plain" and match_plain_interaction_budget else "episodes")
                    if evaluation_interval_episodes and i % evaluation_interval_episodes == 0:
                        test = evaluate_sac(evaluation_env, agent, config, test_episodes)
                        checkpoints.append(_checkpoint(agent, test, i, "validation"))
                final_test = evaluate_sac(evaluation_env, agent, config, test_episodes, 400_000)
                final_summary = phase_summary(final_test)
                outputs[name] = {
                    "warmup": warmup, "warmup_summary": phase_summary(warmup, warmup_updates),
                    "post_warmup_test": initial_test if name == "planned" else None,
                    "post_warmup_test_summary": phase_summary(initial_test) if name == "planned" else None,
                    "post_warmup_policy_status": "trained" if warmup_updates else "no_optimizer_updates",
                    "training": training,
                    "training_summary": phase_summary(training, agent.updates - warmup_updates),
                    "post_training_test": final_test if train_count else None,
                    "post_training_test_summary": final_summary if train_count else None,
                    "test": final_test, "test_summary": final_summary,
                    "evaluation_checkpoints": checkpoints,
                    "pre_test_environment_steps": agent.stored_environment_steps,
                    "optimizer_updates": agent.updates,
                    "final_entropy_coefficient": agent.diagnostics()["entropy_coefficient"],
                    "evaluation_seconds": sum(c["summary"]["elapsed_seconds"] for c in checkpoints)
                                          + final_summary["elapsed_seconds"],
                }
                print("{} final success: {:.3f} | {} interactions | {} updates".format(
                    name, final_summary["success_rate"], agent.stored_environment_steps, agent.updates), flush=True)
                del agent
        finally:
            env.close()
            if evaluation_env is not None:
                evaluation_env.close()
    return {
        "schema_version": 2, "config": config.to_dict(),
        "protocol": {"her_enabled_both_agents": True, "recovery_enabled": False,
                     "deterministic_evaluation": True, "evaluation_learns": False,
                     "action_space": "continuous", "reward": "sparse", "continuing_task": False,
                     "warmup_guidance_mode": planner_guidance_mode,
                     "warmup_action_source": (
                         "planner_only" if planner_guidance_mode == "planner_only"
                         else ("per_action_linear_planner_probability"
                               if planner_guidance_mode == "linear_warmup_anneal"
                               else ("planner_direction_plus_annealed_gaussian_noise"
                                     if planner_guidance_mode == "planner_action_noise"
                                     else "planner_only_with_actor_behavior_cloning"))
                     ),
                     "planner_action_noise_initial_std": planner_action_noise_initial_std,
                     "behavior_cloning": planner_guidance_mode == "planner_behavior_cloning",
                     "behavior_cloning_initial_weight": behavior_cloning_initial_weight,
                     "behavior_cloning_final_weight": behavior_cloning_final_weight,
                     "imitation_loss": False,
                     "budget_unit": "environment_steps" if match_plain_interaction_budget else "episodes",
                     "equal_interaction_budget": match_plain_interaction_budget,
                     "validation_seed_offset": 300_000, "final_test_seed_offset": 400_000,
                     "evaluation_interval_episodes": evaluation_interval_episodes},
        "runtime": dict(python=platform.python_version(), **{
            p: importlib.metadata.version(p) for p in
            ("stable-baselines3", "torch", "gymnasium", "gymnasium-robotics", "numpy")}),
        "planner_warmup_episodes": planner_warmup_episodes,
        "planned_train_episodes": planned_train_episodes,
        "plain_train_episodes": plain_train_episodes,
        "plain_interaction_budget_target": (
            outputs["planned"]["pre_test_environment_steps"]
            if match_plain_interaction_budget else None
        ),
        "test_episodes": test_episodes,
        "elapsed_seconds": time.perf_counter() - started,
        **outputs,
    }


def _checkpoint(agent, test, normal_episodes, stage):
    return {"environment_steps": agent.stored_environment_steps,
            "normal_training_episodes": normal_episodes, "stage": stage,
            "optimizer_updates": agent.updates, "summary": phase_summary(test)}


def _progress(config, phase, completed, total, agent, unit="episodes"):
    if completed % config.progress_every_episodes == 0 or completed == total:
        if unit == "steps":
            print("{}: {} episodes | {}/{} target steps | {} updates".format(
                phase, completed, agent.stored_environment_steps, total, agent.updates), flush=True)
        else:
            print("{}: {}/{} episodes | {} steps | {} updates".format(
                phase, completed, total, agent.stored_environment_steps, agent.updates), flush=True)
