# Planned PointMaze experiments

This project compares planner-initialized and plain reinforcement learning in
PointMaze Open and Medium. Both Double DQN and SAC+HER are supported. The physical
environment remains continuous. The planner uses a discrete grid; DQN uses eight
discrete force directions and SAC uses the native continuous two-dimensional action.

## Architecture

```text
main.py                    Choose and launch an experiment
agents/
  dqn.py                   Network, replay, Double DQN updates, wall recovery
  interface.py             Agent contract and behaviour-preserving DQN adapter
  sac.py                   Stable-Baselines3 SAC adapter; no recovery or epsilon
  replay/point_maze_her.py  HER terminal correction for episodic sparse PointMaze
  execution.py             Scoped CPU thread settings (restored after SAC runs)
  factory.py               Single agent-construction entry point
  exploration.py           Per-agent 75%-of-training epsilon schedule
  evaluation.py            Preserve training state across sweep evaluations
environments/
  common.py                Gymnasium adapter and eight-action table
  point_maze_open.py       Open grid geometry and coordinate mapping
  point_maze_medium.py     Medium wall map, mapping and safe grid neighbours
  registry.py              Gym id → environment, grid and planner specification
configurations/
  point_maze_open.py       Existing Open DQN preset
  point_maze_medium.py     Existing Medium DQN preset
  sac.py                   Independent SAC settings and Open/Medium presets
  common.py                Validation and complete report configuration
planners/
  point_maze_open.py       Open BFS routing and continuous steering
  point_maze_medium.py     Medium BFS routing and continuous steering
experiments/
  runner.py                Shared episode loop and summary metrics
  warmup_sweep.py          One cumulative learner, evaluated at checkpoints
  comparison.py            Planned vs plain training/testing phases
  dqn_multiseed.py         Repeated-seed DQN comparison and aggregate report
  sac_comparison.py        SAC phases, isolated tests and resource accounting
  sac_multiseed.py         Repeated-seed SAC+HER BC/control comparison
  validation.py            Standalone planner checks and fresh warm-up trials
reporting/
  artifacts.py             Unique run directories and JSON writing
  comparison.py            Combined 200-episode-average graph and JSON report
  sac.py                   SAC reward/alpha figure and success vs actual steps
tests/test_refactor.py      Small-budget behavioral and reporting checks
tests/test_sac.py           SAC/HER correctness, isolation and resume tests
```

Both services construct their agents with `agents.create_agent`, their environment
with `config.environment.create`, and their planner with `planners.create_planner`.
The configuration binds the environment and planner to the agent preset.
`run_planned_vs_plain(config, ...)` dispatches by algorithm; the original
`run_planned_vs_plain_dqn` remains available. Both algorithms share the episode
runner and cumulative sweep, but keep their own learning/exploration semantics.

## Running experiments

In `main.py`, choose one of:

- `run_open_warmup_sweep()`
- `run_medium_warmup_sweep()`
- `run_open_comparison()`
- `run_medium_comparison()`

SAC entry points are `run_open_sac_comparison`, `run_medium_sac_comparison`,
`run_open_sac_warmup_sweep`, and `run_medium_sac_warmup_sweep`.

`main()` currently calls `run_pointmaze_suite()`: the complete five-seed replication
suite described below. It is intentionally long-running. To run one study only,
replace that call with its named function:

- `run_open_dqn_pointmaze_study()` — Open DQN, 25 planner warm-up + 675 planned
  / 700 plain training episodes, 5,000 final test episodes.
- `run_medium_dqn_pointmaze_study()` — Medium DQN, 1,000 planner warm-up + 9,000
  planned / 10,000 plain training episodes, 5,000 final test episodes.
- `run_medium_sac_bc_stability_study()` — Medium SAC+HER, five seeds; compares
  Planned SAC+HER+BC (`lambda: 1.0 → 0.4`) with planner-only warm-up and a
  real-interaction-matched plain SAC+HER arm.

Each seed creates an individual JSON report and comparison graph. Each study
also writes one aggregate JSON with seed-level success, real interaction/update
counts, descriptive statistics, and paired planned-minus-plain outcomes.

Each function accepts a configuration override. Comparison functions also accept
independent warm-up, planned-training, plain-training and test counts:

```python
from dataclasses import replace
from configurations import PointMazeMediumConfig
from main import run_medium_comparison, run_medium_warmup_sweep

config = replace(PointMazeMediumConfig(), seed=43)
result, path = run_medium_comparison(
    config=config,
    planner_warmup_episodes=2000,
    planned_train_episodes=8000,
    plain_train_episodes=10000,
    test_episodes=5000,
)

# Run separately if you want to select a warm-up checkpoint.
sweep_config = replace(config, warmup_episode_counts=(250, 500, 1000))
# report, path = run_medium_warmup_sweep(sweep_config)
```

## DQN: what learns, and when?

Planner warm-up selects every action with the planner, while DQN stores those
transitions and learns from replay. This is reward-based Double DQN learning from
demonstrations; it does not add a supervised action-imitation loss.

Normal training uses epsilon-greedy DQN actions. Each agent independently reaches
epsilon 0.05 at 75% of its own normal-training episode budget. Warm-up does not
advance that schedule. Recovery can override agent actions, including when epsilon
is zero; a greedy test therefore measures the network **plus configured recovery**.

The comparison protocol retains the existing sequence and reset seeds:

1. Planned agent: planner warm-up; if normal training is requested, a test after
   warm-up; normal training; final test.
2. Plain agent: normal training; final test on the same tasks as planned's final test.

Warm-up and either normal-training count may be zero. With no planned normal
training, its single final test is the post-warm-up result. The graph labels the
absent post-training phase as N/A.

The cumulative sweep keeps a single agent, optimizer, replay and target network.
For checkpoints (250, 500, 1000), it trains an additional 250, 250 and 500 episodes.
Its evaluations use epsilon 0 and never store transitions or optimize weights.
Random generators and temporary recovery state are restored after the pause;
learning resumes with the same trajectory as uninterrupted planner training.

## SAC+HER protocol

Both arms use **SAC with HER**; only the planned arm receives planner warm-up.
Both start with identically seeded networks and separate replay/optimizers.
No old SAC configuration is reused. The current stability preset uses two
256-unit hidden layers, learning rate 1e-4, batch 256, gamma .99, tau .001,
fixed entropy coefficient 0.1, 200,000 replay transitions, and one gradient
update per four interactions after learning starts (1,000 transitions for Open,
5,000 for Medium). Set `entropy_coefficient="auto"` only for a separately
recorded automatic-temperature ablation.

Planner warm-up executes only existing planner directions, converted to continuous
force vectors, and updates SAC from those experiences. This is off-policy RL,
**not behaviour cloning**. All learning state is retained for normal training.
If the warm-up has fewer transitions than `learning_starts_transitions`, its
post-warm-up test is still an untrained policy; the JSON marks this explicitly.
The replay also requires a completed episode before any update can happen.

### Annealed planner-guidance variant

`planner_guidance_mode="linear_warmup_anneal"` is a distinct planned-SAC method
condition. During the warm-up phase, the probability of using the planner is
linearly reduced from 1.0 in the first warm-up episode to 0.0 in the final one.
The choice is made independently for every action: remaining actions use SAC's
normal training action selection. Normal training after warm-up is fully SAC
controlled. This tests a gradual handover without changing the autonomous final
evaluation policy. It is not equivalent to planner-only warm-up, so reports save
the mode, planned action count and planned-action fraction. `planner_only`
remains the default and reproduces the original protocol.

`planner_guidance_mode="planner_action_noise"` is a separate continuous-action
coverage experiment. The planner chooses every warm-up direction; Gaussian noise
is then added to its two-dimensional force vector and clipped to the environment
action bounds. The noise standard deviation decreases linearly from
`planner_action_noise_initial_std` to zero over warm-up. This does **not** hand
control to an untrained SAC actor. The current pilot uses initial standard
deviation `0.2`; this is reported with the action counts in the result JSON and
graph, and must not be compared as if it were planner-only warm-up.

`planner_guidance_mode="planner_behavior_cloning"` is the direct actor-transfer
variant. Planner-only warm-up actions are also saved to a separate demonstration
buffer. SAC's actor loss receives an additional term
`lambda * MSE(deterministic_actor_action, planner_action)`. The critic loss,
HER buffer and entropy term remain standard SAC. `lambda` is held at the initial
weight during warm-up, then linearly changes to the requested final weight across
normal training. This is a distinct method named **Planned SAC+HER+BC**, not a
hyperparameter adjustment to ordinary planner-only SAC+HER.

During normal training SAC collects uniform random actions until its *total*
collected training transitions reach the learning-start threshold, then samples
its stochastic actor. Warm-up transitions count toward that threshold. SAC has
no epsilon or epsilon decay. Recovery is disabled; the actor may legitimately
choose small/zero forces. DQN recovery and its learning logic are unchanged.

HER uses `future` goals from the same completed episode, with approximately 80%
virtual samples in each minibatch. Rewards are recomputed by the environment.
Original observations/actions are not overwritten, and virtual experiences add
no environment steps. Our `EpisodicPointMazeHER` extension also recomputes the
terminal mask: for the supported sparse episodic PointMaze task, reward 1 means
the relabelled goal has been reached. Original time-limit truncations permit
bootstrapping; virtual success is terminal even at the time limit. This extension
deliberately rejects reward normalization and is not a general-purpose HER buffer
for other task semantics. The integration is tested against SB3 **2.7.1**.

SAC evaluations use deterministic actor output in a **separate environment**.
They insert no transitions and update neither actor, critics, targets, optimizers,
nor entropy coefficient. RNG and module modes are restored. Tests verify that
training after a pause matches uninterrupted training exactly.

The planned arm always has a diagnostic test after warm-up (an untrained baseline
when warm-up=0), and a held-out final test. With zero normal training, the final
test is a second evaluation of the same checkpoint; post-training fields are N/A.
Plain also has an initial untrained baseline. Optional periodic validation tests
use matched task seeds; final tests use a separate matched set. Evaluation time
and interactions do not count as training, and are reported separately.

```python
from dataclasses import replace
from configurations import PointMazeMediumSACConfig
from main import run_medium_sac_comparison, run_medium_sac_warmup_sweep

config = replace(PointMazeMediumSACConfig(), seed=43)
# Run one or the other, not both unless intended:
result, path = run_medium_sac_comparison(
    config, planner_warmup_episodes=40, planned_train_episodes=20,
    plain_train_episodes=60, test_episodes=50, evaluation_interval_episodes=10,
)
# report, path = run_medium_sac_warmup_sweep(replace(
#     config, warmup_episode_counts=(50, 100, 200), evaluation_episodes=500))
```

The sweep uses one SAC agent and cumulative checkpoint counts just like DQN.
Only compact configuration and warm-up/evaluation summaries are saved.

## New output layout

```text
resources/planned-maze-point/
  pointmaze_open/ or pointmaze_medium/
    dqn/ or sac_her/
      warmup_sweep/<unique-run-id>/summary.json
      comparison/<unique-run-id>/result.json
                                  comparison.png
                                  success_by_interactions.png  (SAC only)
      pointmaze_open_dqn/<unique-run-id>/aggregate.json
      pointmaze_medium_dqn/<unique-run-id>/aggregate.json
      bc_multiseed/<unique-run-id>/aggregate.json
```

Every execution has a unique directory so changing recovery settings or repeating
a seed does not overwrite another run. Older files remain at their original paths.

The sweep produces **one aggregate JSON** containing the full configuration and
a chronological checkpoint list. Each checkpoint contains cumulative warm-up and
evaluation summaries: success, mean reward, environment steps, recovery counts,
warm-up time and update counts. It contains no per-episode arrays. The same report
is updated after each completed checkpoint, so completed results survive an
interruption. The best observed count is a descriptive selection on this one
evaluation set; validate it on independent seeds/tasks before thesis conclusions.

The comparison keeps detailed phase histories and summaries in JSON, plus **one
combined graph**:

- Planned rewards in blue; plain rewards in orange; planner warm-up in grey.
- A trailing **200-episode mean**; early episodes use available samples.
- Smoothing resets at the warm-up/training boundary so planner rewards do not
  inflate the network's first training points.
- A red boundary marks warm-up completion.
- Plain training and its epsilon are shifted right by the warm-up count. Both
  normal-training curves start at warm-up + 1.
- Dashed epsilon curves use matching agent colours and a right-hand axis.
- A box shows planned post-warm-up, planned post-training, and plain final-test
  success rates.

The shifted axis aligns **normal-training starts**. Equal x coordinates do not
imply equal interactions or compute; actual episode counts, environment steps and
optimizer updates remain in the JSON. Each epsilon uses its own training budget.
When schedules coincide, the dashed lines overlap because the values are equal.

For SAC, the combined figure has a **separate alpha panel instead of epsilon**.
Alpha is the learned entropy coefficient, not a probability, and need not
decrease monotonically. The second graph plots deterministic validation success
against actual cumulative training interactions, **without shifting plain's
origin**. Held-out final test results are marked separately. With no periodic
tests requested, this graph has only baseline/post-warm-up and final points;
it must not be interpreted as a densely measured learning curve.

SAC JSON also records dependency versions, phase timing, planner-action selection
time (a subset of warm-up time), actual training steps, update counts and all
evaluation checkpoint summaries. Equal episode counts are not equal step or
compute budgets. Use matched interaction budgets and multiple independent seeds
before drawing thesis conclusions; select warm-up on validation runs, not on the
held-out final test results.

### Dependencies

SAC requires `stable-baselines3==2.7.1` (added to `requirements.txt`). The existing
virtual environment already provides it; no package installation or environment
upgrade is needed for the verification runs. It currently has Python 3.9.6,
Gymnasium 1.1.1, Robotics 1.3.1 and Torch 2.8.0. Note that the pre-existing
requirements file specifies Robotics 1.4.2: that pin has not been silently changed
or installed. Each SAC comparison records the versions actually used.

## Replot an existing comparison without training

```python
from reporting.comparison import replot_saved_comparison

path = replot_saved_comparison("/absolute/path/to/old-comparison.json")
print(path)
```

This reads the configuration and results recorded in that file and writes a new
comparison directory; the original report is preserved.

## Planner validation and regression checks

```python
from configurations import PointMazeMediumConfig
from experiments.validation import run_planner_validation

validation = run_planner_validation(PointMazeMediumConfig(), episodes=100)
print(validation["summary"])
```

Open's neighbour ordering and steering are unchanged. Medium's wall map,
coordinate mapping and no-corner-cutting rule live in its environment module;
the planner receives that grid. Neither planner requires DQN's recovery.

From this directory, run:

```sh
python -m unittest discover -s tests -v
```

These checks use small training budgets and temporary outputs. They cover
evaluation isolation, cumulative sweep counts, geometry against the installed
Gymnasium maze, graph alignment and smoothing, zero-length phases, and output
collision prevention.
