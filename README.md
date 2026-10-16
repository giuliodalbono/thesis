# Planning-Guided Reinforcement Learning

Research code for a master's thesis in Computer Science Engineering exploring whether deterministic planning on simplified models can reduce the cost of training reinforcement-learning agents while preserving their performance in the original environment.

The central question is how to transfer useful decisions from an abstract model to a learned policy. The experiments progress from a discrete stochastic navigation problem, FrozenLake, to continuous navigation in PointMaze. They compare autonomous learning with planner-generated experience and, in the latest experiments, explicit imitation of planner actions.

This is an experimental repository: protocols, controls, and measurement conventions differ between experiment families. The distinctions below are part of interpreting the results. Multiple abstraction layers are a research direction; the implementations here use a single planning abstraction.

## Experiment overview

| Experiment | Real task and planning abstraction | Agents / comparisons | Main question |
| --- | --- | --- | --- |
| FrozenLake DQN | Slippery grid navigation; deterministic nominal transitions for planning | Planner-guided DQN versus plain DQN | Can demonstrations from a simplified transition model help learning under stochastic dynamics? |
| FrozenLake PPO | Same slippery maps, without planning | PPO baseline | How does an on-policy learner behave as map size increases? |
| FrozenLake SAC | Same slippery maps, with continuous actions quantized into grid moves | SAC baseline, without HER or planning | How does a continuous-policy learner behave through a discrete-action adapter? |
| PointMaze Open DQN | Continuous motion mapped to a free-space grid | Planned versus plain Double DQN | Does planner warm-up help on a simple continuous navigation task? |
| PointMaze Medium DQN | Continuous motion mapped to a grid containing walls | Planned versus plain Double DQN | Does the same transfer mechanism remain useful with obstacles? |
| PointMaze SAC+HER | Continuous goal-conditioned control with the same grid planner | Planner-only warm-up, guidance/noise variants, BC variant, and plain SAC+HER | How can an actor learn from and retain planner demonstrations? |
| PointMaze warm-up sweeps | Open or Medium; DQN or SAC+HER | One learner evaluated at cumulative demonstration budgets | How much planner experience produces a useful autonomous policy? |

## Repository layout

```text
planned-deterministic-frozen-lake/
  main.py                  Map selection, experiment switches and budgets
  planner.py               Unified Planning / Fast Downward model
  dqn_agent.py             DQN, planner guidance and checkpoints
  ppo_agent.py             Planner-free PPO baseline
  sac_agent.py             Planner-free SAC with action quantization
  slippery_probability_wrapper.py
  run_from_checkpoint.py   Evaluate or continue a saved DQN
  plot_utility.py
planned-maze-point/
  main.py                  Individual experiments and the PointMaze suite
  agents/                  DQN, SAC, HER replay, BC and evaluation state handling
  environments/            Environment adapters, grid geometry and action mapping
  configurations/          Agent/environment presets
  planners/                Open and Medium grid planners
  experiments/             Comparisons, sweeps, multi-seed studies and validation
  reporting/               JSON reports and comparison figures
  tests/                   Small-budget regression checks
resources/                 Experiment reports, figures and FrozenLake checkpoints
```

## 1. FrozenLake: planning with a simplified transition model

The FrozenLake experiments use hand-written maps from **4×4 to 17×17**. The agent must reach the goal without falling into a hole. Rewards remain sparse: reaching the goal yields a reward of one.

Despite the directory name `planned-deterministic-frozen-lake`, the current environment is stochastic. A wrapper changes its transition probabilities to **0.8 for the requested direction and 0.1 for each perpendicular direction**. The deterministic component is the planner's model: it assumes that the requested move succeeds.

The planner represents cells as locations, the agent's position as a fluent, and movement as actions. It uses Unified Planning with the Fast Downward backend. Hole and goal cells have no outgoing planning actions, so a successful plan cannot pass through a hole. At execution time, the action is selected from the current observed state; plans are cached by state, allowing the controller to react after a slip rather than blindly execute an entire nominal route.

### Planner-guided DQN

DQN observes a one-hot encoding of the current cell. During planner warm-up, experience is collected through planner actions and used for ordinary replay-based Q-learning. There is no supervised imitation loss. If planning fails, action selection falls back to a random move.

The action-selection mechanism has two probabilities:

- `epsilon` selects exploration instead of the greedy network action.
- `epsilon_2` selects the planner instead of a random action within exploration.

Both are one during the configured planner warm-up. In the current normal-training configuration, planned DQN starts at `epsilon=0.5`, plain DQN starts at `epsilon=1.0`, and `epsilon_2=0`, so planner calls cease after warm-up. Episode-based exponential schedules control subsequent exploration. Evaluation sets both probabilities to zero and performs no learning.

These differences matter for comparison: warm-up episodes are **additional** to the requested normal-training budget, and the two arms start normal training with different exploration rates. This protocol therefore measures the combined effect of demonstrations and its exploration configuration; it does not isolate demonstrations under equal interaction budgets.

The runner supports post-warm-up evaluation and saves DQN checkpoints after warm-up and final training. `run_from_checkpoint.py` can evaluate a checkpoint or continue training from a planner-warm-up checkpoint. Historical figures include different warm-up and training budgets; current constants should not be treated as the configuration of every archived figure.

### PPO and SAC baselines

PPO uses Stable-Baselines3 with one-hot observations and the native discrete action space. It is planner-free. Map-size presets adjust the training-step budget and optimization settings.

FrozenLake SAC is also planner-free. Because Stable-Baselines3 SAC expects continuous actions, the adapter exposes a scalar action in `[-1, 1]` and quantizes it into the four grid actions. This is a substantive adaptation: it is neither native discrete SAC nor the continuous two-dimensional controller used in PointMaze. It does not use HER. PPO and SAC have timestep budgets rather than the DQN runner's episode budgets, so their figures should not be interpreted as resource-matched algorithm comparisons.

## 2. PointMaze: planning over a continuous environment

PointMaze uses Gymnasium Robotics environments **`PointMaze_Open-v3`** and **`PointMaze_Medium-v3`** with episodic goal reaching. The environment retains continuous position, velocity and dynamics. Only the planner's representation is a discrete grid; the neural agents continue to observe continuous state and goal information.

The Open and Medium planners use breadth-first search on their environment-specific grids and steer toward successive waypoints. The Medium grid includes walls and prevents diagonal corner cutting. Steering translates waypoint directions into force actions and includes a near-target braking rule. Thus demonstrations combine a known map, deterministic routing, and a hand-written motion controller. They are not trajectories produced by an uninformed learner.

The experiment sequence is:

1. Collect planner-controlled warm-up episodes while updating the planned agent.
2. Evaluate the planned policy after warm-up.
3. Continue autonomous training with the learned agent.
4. Evaluate planned and plain policies on matched final-test tasks.

The plain agent starts from independently instantiated, identically seeded network weights and receives no planner experience. Warm-up and normal-training budgets can be configured independently, including zero-length phases.

### Double DQN: Open and Medium

DQN uses **eight discrete force directions**: four cardinal and four normalized diagonal actions. The no-op action is excluded. Observations concatenate the physical observation, achieved goal and desired goal.

Planner warm-up populates replay and updates the Q-network through reward-based learning. Normal training uses epsilon-greedy selection. Each agent independently anneals epsilon linearly from `1.0` to `0.05` over the first 75% of its normal-training episode budget. Warm-up does not advance this schedule.

The implementation uses Double DQN and a target network. It also has an explicit **wall-recovery controller**: insufficient progress triggers opposite-direction actions followed by random actions. Recovery applies to both planned and plain agents and remains available during evaluation. Consequently, epsilon-zero results measure the greedy network **with recovery**, including its random recovery actions, rather than a purely deterministic network policy. Removing the no-op and adding recovery are changes to the control protocol that must be considered when interpreting performance.

Open tests the transfer mechanism with simple geometry. Medium introduces obstacle-constrained routes and larger training budgets. Neither DQN condition uses HER or behavior cloning.

### SAC with Hindsight Experience Replay

PointMaze SAC uses Stable-Baselines3 SAC with the **native continuous two-dimensional action space**. Both planned and plain SAC use Hindsight Experience Replay (HER), which relabels sampled transitions with goals achieved later in the same completed trajectory. With four sampled goals configured, approximately 80% of a replay minibatch consists of virtual-goal samples. These samples do not add real environment interactions.

The repository includes an episodic PointMaze HER correction: relabelled successful transitions are terminal, while genuine time-limit truncations retain bootstrapping semantics. Rewards are recomputed by the environment. This replay implementation is specific to the supported sparse, episodic task and is not a general HER replacement.

Collection is controlled by the experiment runner so planner actions can be inserted into SAC replay directly. Learning begins only after the configured interaction threshold and a completed trajectory. Normal SAC collection uses random actions before that threshold and stochastic actor actions afterward; warm-up interactions count toward the threshold.

Recovery is disabled for SAC. Tests use deterministic actor output, in a separate environment, without replay insertion or optimization. Evaluation preserves training RNG state and module modes.

The current stability preset fixes the entropy coefficient **alpha at `0.1`**. Alpha weights SAC's entropy objective; it is not epsilon, and a fixed coefficient does not imply a fixed amount of action randomness. Automatic entropy tuning is supported as a separate setting. The selected suite uses one update event per eight interactions; the general SAC preset defaults to four. Fixed alpha is an experimental choice, not an established optimum.

### SAC planner-transfer variants

| Mode | Warm-up action source | Learning change | After warm-up |
| --- | --- | --- | --- |
| `planner_only` | Planner | Standard SAC+HER updates from demonstrated experience | Autonomous SAC+HER |
| `linear_warmup_anneal` | Per-action planner probability decreases from 1 to 0 | Same SAC+HER objective | Autonomous SAC+HER |
| `planner_action_noise` | Planner directions plus clipped Gaussian noise, annealed toward zero | Same SAC+HER objective, broader collected actions | Autonomous SAC+HER |
| `planner_behavior_cloning` | Planner; state/action examples also stored separately | Additional actor imitation loss | Autonomous collection with continued imitation from stored demonstrations |

**Behavior cloning (BC)** adds the following term to SAC's actor objective:

```text
actor_loss = SAC_actor_loss + lambda × MSE(deterministic_actor_action, planner_action)
```

Only planner examples enter the separate demonstration buffer. The coefficient is held at its initial value during warm-up and then linearly changes toward its requested final value during normal training. The selected protocol uses `1.0 → 0.4`. No planner is consulted for autonomous action selection or evaluation; the actor still receives a learning signal from stored planner examples.

This condition is called **Planned SAC+HER+BC** because it changes the learning objective. The `planner_only` control has no BC loss at any phase. It is different from a BC experiment that imitates during warm-up and then anneals the BC coefficient to zero.

Exploratory work tested gradual planner handover, planner-action noise, and BC endpoints of `0.0`, `0.2`, `0.3`, `0.4`, `0.6`, and `1.0`. These experiments motivated the selected retention protocol; they do not establish a universal stability threshold.

### Cumulative warm-up sweeps

Sweeps are available for Open and Medium with either DQN or SAC+HER. A single agent retains its network, replay and optimizer between cumulative checkpoints. For `(250, 500, 1000)`, collection proceeds for 250 episodes, then 250 more, then 500 more, with evaluation at each checkpoint.

Evaluation does not update weights or replay, and temporary evaluation state is restored before collection resumes. The current sweep studies planner-only experience; its results should not be assumed to apply directly to the BC comparison protocol. Warm-up selection is exploratory and should be validated using independent training seeds and evaluation tasks.

## PointMaze suite and current observations

`run_pointmaze_suite()` runs the following studies over seeds **42–46**:

| Study | Planner warm-up | Planned normal training | Plain normal training | Episodes per evaluation |
| --- | ---: | ---: | --- | ---: |
| Open DQN | 25 | 675 | 700 episodes | 5,000 |
| Medium DQN | 1,000 | 9,000 | 10,000 episodes | 5,000 |
| Medium SAC+HER+BC | 400 | 50 | Match planned real training steps | 200 |
| Medium SAC+HER planner-only control | 400 | 50 | Match this control's real training steps | 200 |

The SAC study runs a separate plain baseline for each planned condition. It finishes the last plain episode after reaching the target, so actual steps can overshoot by less than one maximum episode length. This is approximate interaction matching, not exact matching of computation or episode count. DQN matches total episode counts, including warm-up, and reports the resulting unequal interaction counts.

The October 2026 suite produced the following final success rates (mean ± sample standard deviation across five training seeds):

| Study | Planned | Plain |
| --- | ---: | ---: |
| Open DQN | 90.45% ± 3.88% | 85.71% ± 5.80% |
| Medium DQN | 53.23% ± 5.08% | 60.60% ± 12.89% |
| Medium SAC+HER+BC | 75.90% ± 8.47% | 8.10% ± 3.19% |
| Medium SAC+HER planner-only control | 9.00% ± 3.39% | 8.70% ± 2.97% |

These results suggest that planner warm-up alone has mixed value, while explicit actor imitation substantially helps the short-budget Medium SAC experiment. They do not show that abstraction alone accounts for the BC advantage, or that plain SAC cannot succeed with longer training. Open DQN wins on three of five seeds; Medium DQN wins on one. Small average DQN differences should not be described as established improvements without uncertainty analysis.

BC also does not completely prevent degradation: in the suite, seed 44 fell from 87% after warm-up to 63% after autonomous training. The five seeds were reused from development, so this suite is a replication of the selected protocol, not independent confirmation on unseen seeds. Rerunning identical seeds is useful for reproducibility but does not increase the number of independent trials.

The corresponding aggregate reports are under [PointMaze resources](resources/planned-maze-point/). Historical report paths may reference an earlier checkout or the previous `planned-maze-point-v2` directory name; locate the same run ID in the current resource tree if an absolute path no longer resolves.

## Installation and running

Use a virtual environment and run commands from the repository root unless otherwise noted:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r planned-maze-point/requirements.txt
```

The PointMaze requirements pin Gymnasium Robotics to `1.4.2` and Stable-Baselines3 to `2.7.1`; other dependencies are not fully locked. Earlier runs used Robotics `1.3.1`, so installing the current requirements is not an exact reconstruction of every historical environment. SAC reports record runtime package versions. MuJoCo support is needed for PointMaze, and rendering may require platform-specific graphics setup.

FrozenLake planning additionally needs Unified Planning and its Fast Downward backend:

```sh
python -m pip install unified-planning up-fast-downward
```

### Run PointMaze

```sh
cd planned-maze-point
python main.py
```

The default `main()` launches the **full suite**, including long Medium DQN runs. To select one study, replace its call with `run_open_dqn_pointmaze_study()`, `run_medium_dqn_pointmaze_study()`, or `run_medium_sac_bc_stability_study()`.

Individual comparisons and sweeps are callable without running the suite. For example, from the PointMaze directory:

```python
from dataclasses import replace
from configurations import PointMazeOpenConfig
from main import run_open_comparison, run_open_warmup_sweep

config = replace(PointMazeOpenConfig(), seed=43)
result, path = run_open_comparison(
    config=config,
    planner_warmup_episodes=25,
    planned_train_episodes=675,
    plain_train_episodes=700,
    test_episodes=5000,
)

# Optional separate sweep:
# run_open_warmup_sweep(replace(config, warmup_episode_counts=(25, 50, 100)))
```

Equivalent entry points exist for Medium DQN and Open/Medium SAC. Configuration classes bind agent settings to the appropriate environment and planner. Importing PointMaze `main` does not launch training.

### Run FrozenLake

Edit `planned-deterministic-frozen-lake/main.py` to choose `RUN_DQN`, `RUN_PPO`, `RUN_SAC`, `SELECTED_MAPS`, and the budgets, then run:

```sh
cd planned-deterministic-frozen-lake
python main.py
```

The current switches select SAC across all configured maps. DQN planner warm-up is controlled by `ENABLE_PLANNER_WARM_UP` and `PLANNER_WARM_UP_EPISODES`. PPO and SAC use map-size presets. To use saved DQN checkpoints, configure `CHECKPOINT_PATH` and `MODE` in `run_from_checkpoint.py` before running that script.

## Reports and interpretation

PointMaze reports live under `resources/planned-maze-point/<environment>/<agent>/`:

- `comparison/<run-id>/result.json` contains complete configuration, phase histories, test summaries, training steps and update counts. SAC also records runtime versions and timing.
- `comparison/<run-id>/comparison.png` shows planned and plain rewards, the warm-up boundary, exploration information, and test success percentages.
- SAC's `success_by_interactions.png` plots available evaluations against real training interactions.
- `warmup_sweep/<run-id>/summary.json` stores cumulative warm-up/evaluation summaries.
- `pointmaze_open_dqn/`, `pointmaze_medium_dqn/`, and `bc_multiseed/` contain study aggregates.
- `pointmaze_medium/sac_her/pointmaze_suite/<run-id>/suite_index.json`, relative to the PointMaze resource root, links the three study aggregates.

Reward curves use a trailing mean of up to 200 episodes. Smoothing restarts at the warm-up boundary so successful planner rewards do not inflate the autonomous training curve. Plain training is shifted right by the warm-up episode count to align normal-training starts. This display alignment does not represent equal interactions. DQN plots epsilon; SAC plots alpha, which is flat for the current fixed-alpha preset. Tests are summarized separately from training rewards.

Each completed PointMaze comparison is saved separately. A study aggregate is written when that study finishes, and the suite index is written after the full suite finishes. An interruption preserves already saved comparisons but does not automatically resume an unfinished agent or seed. Sweeps update their compact report after each completed checkpoint.

FrozenLake outputs use `resources/planned-deterministic-frozen-lake/graphs/` and `checkpoints/`, with separate PPO/SAC graph subfolders. These older artifacts use descriptive filenames rather than PointMaze's unique run directories; rerunning the same named output may replace it.

For scientific comparisons, distinguish training episodes, real environment interactions, gradient updates, and wall-clock time. Include planner computation and demonstration collection when assessing cost. BC adds computation beyond plain SAC. Evaluation interactions are separate from training interactions, and different experiment families currently use different budgets and evaluation sizes. A higher reward during planner-controlled warm-up measures planner performance, not autonomous policy quality.

## Validation and development

From `planned-maze-point/`:

```sh
python -m unittest discover -s tests -v
```

The tests use small budgets and temporary outputs. They cover DQN compatibility, environment/grid geometry, frozen evaluation and resumed learning, cumulative sweeps, HER relabelling, planner action collection, BC behavior, and graph alignment. The FrozenLake transition wrapper has its own checks:

```sh
cd planned-deterministic-frozen-lake
python -m unittest slippery_probability_wrapper
```

Changes to action mappings, recovery, reward/terminal handling, planner access, replay, exploration schedules, or budget matching should be treated as protocol changes and described in the resulting reports. Keep new comparisons identifiable by their environment, agent, seed, configuration, and method condition. Bug reports and experimental extensions are most useful when accompanied by a minimal configuration and the relevant saved report.
