# Planner-only PointMaze warm-up sweep

This workspace estimates how many planner demonstrations a Double-DQN agent
needs before ordinary DQN training is introduced in a later experiment. Open
and Medium PointMaze configurations are available separately.

For every configured candidate warm-up count, one identically seeded DQN keeps
learning planner transitions cumulatively and performs these phases:

1. Continue the deterministic abstract planner until the next cumulative
   warm-up checkpoint. Every planner transition enters replay and trains the
   Double-DQN network.
2. Pause for 5,000 greedy DQN evaluation episodes. These episodes do not enter
   replay, do not update the network, and do not advance the agent RNG state.

There are no alternating planner/agent pairs, plain-DQN baselines, or ordinary
post-warm-up training in this sweep. The DQN configuration remains unchanged:
eight movement actions, Double DQN, replay, target network, and the shared
wall-stuck recovery controller during greedy evaluations.

The Open configuration uses 25 to 2,000 planner-only episodes; Medium uses
250 to 5,000. Each checkpoint writes a detailed JSON result; the final sweep
report ranks candidates by their 5,000-episode greedy-test success. It is a
one-seed estimate only. Freeze the selected count and validate it on separate
seeds before using it in a planned-versus-classical DQN experiment.

The sweep is exposed as `run_planner_only_warmup_sweep()` in
`warmup_sweep.py`. `main.py` is only a launcher, so later experiments can
import and call the sweep without putting experiment logic in the entry point.

## PointMaze Medium support

`PointMaze_Medium-v3` is supported through `PointMazeMediumConfig` in
`point_maze_medium_config.py`. It provides a larger initial pilot configuration: a 1,000-step
episode limit, larger replay/training thresholds, and warm-up candidates from
250 to 5,000 episodes. `PointMazeOpenConfig` in
`point_maze_open_config.py` retains the Open configuration.

The planned experiment automatically selects `PointMazeMediumPlanner` for
the Medium environment. Its abstract graph contains only free cells and blocks
diagonal corner-cutting moves, so paths avoid the maze walls. Before training,
validate its physical execution with:

```python
from point_maze_medium_config import PointMazeMediumConfig
from experiment import run_planner_validation, save_planner_validation

config = PointMazeMediumConfig()
validation = run_planner_validation(config, episodes=100)
print(validation["summary"])
save_planner_validation(config, validation)
```

The planner should achieve a near-100% validation success rate before its
trajectories are used as warm-up supervision. Result filenames and plots now
include the selected environment name, so Open and Medium artifacts do not mix.

For the Medium planner-only warm-up sweep, call
`run_medium_planner_only_warmup_sweep()` from `warmup_sweep.py`. It uses one
network cumulatively: it pauses at each candidate, evaluates greedily, then
continues warm-up from that same learned network. Evaluations never update the
network or replay buffer, and preserve the agent RNG state. The generic
`run_planner_only_warmup_sweep(PointMazeMediumConfig())` is equivalent.

## Planned versus plain DQN service

`planned_vs_plain_dqn.py` exposes `run_planned_vs_plain_dqn()`. It supports a
planner-only warm-up phase, ordinary epsilon-greedy DQN training, and a greedy
test phase. Planner warm-up, planned normal-training, and plain
normal-training counts may all be zero and can be chosen independently. The
planned and plain agents always receive matched test task seeds; only the
planned agent receives planner trajectories first. When
`planned_train_episodes > 0`, the saved planned result contains a greedy
`post_warmup_test` before normal training and a second
`post_training_test` afterward. The final planned `test` is retained as an
alias of that second test for direct comparison with the plain DQN final test.

During normal training, each agent's epsilon is annealed linearly from
`epsilon_start` to `epsilon_final` over the first 75% of *that agent's own*
training-episode budget. Thus, different planned and plain training budgets
receive separate decay horizons; planner-only warm-up does not consume the
planned agent's epsilon schedule.

The separate planner-only warm-up sweep has no normal DQN training phase, so
the bounded epsilon schedule is deliberately not applied there: the planner
selects all warm-up actions and the 5,000-episode evaluation is greedy. Its JSON
trial and aggregate report record this explicitly.

Calling `save_planned_vs_plain_result(config, result)` also writes two PNG
reward curves under `resources/planned-maze-point-v2/plots/` (or the configured
output directory):

- `planned_dqn_training_*.png` combines the planner warm-up and ordinary DQN
  training rewards on one episode axis. A dashed vertical red line marks the
  end of warm-up and start of normal training.
- `plain_dqn_training_*.png` shows the ordinary-training reward curve for the
  plain DQN.

Each image shows the raw per-episode rewards faintly and a 25-episode moving
average prominently, so the learning trend is readable without discarding the
underlying episode data. A secondary purple y-axis displays the epsilon schedule
used during normal DQN training. Planner-only warm-up has no epsilon value,
because the planner—not the DQN policy—selects its actions.
