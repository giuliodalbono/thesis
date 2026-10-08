# Planned PointMaze experiments

This project compares DQN trained with planner demonstrations against plain DQN
in PointMaze Open and Medium. The physical environment remains continuous.
The planner uses a discrete grid; the DQN uses continuous observations and eight
discrete force directions.

## Architecture

```text
main.py                    Choose and launch an experiment
agents/
  dqn.py                   Network, replay, Double DQN updates, wall recovery
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
  common.py                Validation and complete report configuration
planners/
  point_maze_open.py       Open BFS routing and continuous steering
  point_maze_medium.py     Medium BFS routing and continuous steering
experiments/
  runner.py                Shared episode loop and summary metrics
  warmup_sweep.py          One cumulative learner, evaluated at checkpoints
  comparison.py            Planned vs plain training/testing phases
  validation.py            Standalone planner checks and fresh warm-up trials
reporting/
  artifacts.py             Unique run directories and JSON writing
  comparison.py            Combined 200-episode-average graph and JSON report
tests/test_refactor.py      Small-budget behavioral and reporting checks
```

Both services construct their agents with `agents.create_agent`, their environment
with `config.environment.create`, and their planner with `planners.create_planner`.
The configuration binds the environment and planner to the agent preset. A future
agent implementation belongs in `agents/`, with its creation dispatched by the
factory; the current experiment services implement the DQN protocol.

## Running experiments

In `main.py`, choose one of:

- `run_open_warmup_sweep()`
- `run_medium_warmup_sweep()`
- `run_open_comparison()`
- `run_medium_comparison()`

Running main launches that full experiment.

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

## What learns, and when?

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

## New output layout

```text
resources/planned-maze-point-v2/
  pointmaze_open/ or pointmaze_medium/
    dqn/
      warmup_sweep/<unique-run-id>/summary.json
      comparison/<unique-run-id>/result.json
                                  comparison.png
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
