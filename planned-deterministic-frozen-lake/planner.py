from unified_planning.shortcuts import *
from unified_planning.engines import PlanGenerationResultStatus


def define_problem(env, current_state):
    problem = Problem("frozenlake")

    n_states = env.observation_space.n

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
            # Take just first tuple (deterministic version because of no slippery, so just one next state possible if
            # taking action a)
            prob, next_state, reward, done = transitions[s][a][0]

            # Skip impossible actions
            if prob == 0:
                continue

            action_name = f"move_{s}_{next_state}_{a}"
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
            actions = extract_plan(plan)

            print("\nPlan found:")
            for i, act in enumerate(actions):
                print(f"{i}: {act}")

            return actions
        else:
            print("\nPlan not found")
            exit(1)
