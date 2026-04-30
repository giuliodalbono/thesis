from unified_planning.shortcuts import *
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.environment import get_environment

get_environment().credits_stream = None


def most_likely_transition(outcomes, nominal_preferred=None):
    state_probabilities = {}

    for prob, next_state, _, _ in outcomes:
        if prob == 0:
            continue

        if next_state not in state_probabilities:
            state_probabilities[next_state] = 0.0
        state_probabilities[next_state] += prob

    if not state_probabilities:
        return None

    max_prob = max(state_probabilities.values())
    best_states = {s for s, p in state_probabilities.items() if p == max_prob}

    if nominal_preferred in best_states:
        return nominal_preferred

    # Deterministic fallback to keep planner behavior stable.
    return min(best_states)


def _nominal_next_state(s: int, a: int, nrow: int, ncol: int) -> int:
    # Deterministic nominal movement: LEFT, DOWN, RIGHT, UP.
    row = s // ncol
    col = s % ncol

    if a == 0:  # LEFT
        col = max(0, col - 1)
    elif a == 1:  # DOWN
        row = min(nrow - 1, row + 1)
    elif a == 2:  # RIGHT
        col = min(ncol - 1, col + 1)
    elif a == 3:  # UP
        row = max(0, row - 1)

    return row * ncol + col


def define_problem(env, current_state):
    problem = Problem("FrozenLake-v1")

    n_states = env.observation_space.n
    n_row = int(env.unwrapped.nrow)
    n_col = int(env.unwrapped.ncol)

    # Type (represents the cell of the gridmap)
    Location: Type = UserType("Location")

    # Objects (states, of the planner, represent cells of the gridmap)
    locations = [Object(f"l{i}", Location) for i in range(n_states)]
    problem.add_objects(locations)

    # Fluent (at(l) = True, if agent is at Location l)
    at = Fluent("at", BoolType(), l=Location)
    problem.add_fluent(at)

    # Initial State (for all locations l, at(l) = False, except for location of the agent in the current state)
    for i, loc in enumerate(locations):
        problem.set_initial_value(at(loc), i == current_state)

    # Goal (cell G)
    goal_state = n_states - 1
    problem.add_goal(at(locations[goal_state]))

    # Valid transitions from environment
    # transitions[s][a] contains  all possible tuples (prob, next_state, reward, done) for the state s and action a
    transitions = env.unwrapped.P
    # For all states
    for s in range(n_states):
        # For all actions possible
        for a in range(4):
            nominal = _nominal_next_state(s, a, n_row, n_col)
            best_transition = most_likely_transition(
                transitions[s][a],
                nominal_preferred=nominal,
            )
            if best_transition is None:
                continue

            next_state = best_transition
            action_name = f"move_{s}_{a}"
            action = InstantaneousAction(action_name)

            from_l = locations[s]
            to_l = locations[next_state]

            action.add_precondition(at(from_l))
            action.add_effect(at(from_l), False)
            action.add_effect(at(to_l), True)

            problem.add_action(action)

    return problem


def extract_plan(plan):
    return [str(a) for a in plan.actions]


def build_plan(problem):
    with OneshotPlanner(name="fast-downward") as planner:
        result = planner.solve(problem)

        if result.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
            plan = result.plan
            return extract_plan(plan)
        return None
