import unittest
from typing import Tuple

import gymnasium as gym
from gymnasium import Wrapper


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
ACTIONS = (LEFT, DOWN, RIGHT, UP)
EPS = 1e-6


class SlipperyProbabilityWrapper(Wrapper):
    # FrozenLake wrapper that rewrites transition probabilities in `P`.

    def __init__(self, env, p_nominal: float = 0.8, p_slip: float = 0.1):
        super().__init__(env)
        self.p_nominal = float(p_nominal)
        self.p_slip = float(p_slip)
        self._validate_probabilities()
        self._modify_probabilities()

    def _validate_probabilities(self) -> None:
        if self.p_nominal < 0 or self.p_slip < 0:
            raise ValueError("Probabilities must be non-negative.")
        if abs((self.p_nominal + 2 * self.p_slip) - 1.0) > EPS:
            raise ValueError("Expected p_nominal + 2 * p_slip == 1.0.")

    def _next_state_data(self, state: int, action: int) -> Tuple[int, float, bool]:
        inner_env = self.env.unwrapped
        n_row, n_col = inner_env.nrow, inner_env.ncol

        row, col = state // n_col, state % n_col
        if action == LEFT:
            col = max(0, col - 1)
        elif action == DOWN:
            row = min(n_row - 1, row + 1)
        elif action == RIGHT:
            col = min(n_col - 1, col + 1)
        elif action == UP:
            row = max(0, row - 1)

        next_state = row * n_col + col
        tile = inner_env.desc[row, col]
        reward = float(tile == b"G")
        terminated = tile in b"GH"
        return next_state, reward, terminated

    @staticmethod
    def _perpendicular_actions(action: int) -> Tuple[int, int]:
        # Horizontal actions slip to vertical ones, and vice versa.
        return (DOWN, UP) if action in (LEFT, RIGHT) else (LEFT, RIGHT)

    def _modify_probabilities(self) -> None:
        inner_env = self.env.unwrapped
        n_states = inner_env.nrow * inner_env.ncol

        for state in range(n_states):
            for action in ACTIONS:
                nominal = self._next_state_data(state, action)
                slip_a, slip_b = self._perpendicular_actions(action)
                slip_1 = self._next_state_data(state, slip_a)
                slip_2 = self._next_state_data(state, slip_b)

                inner_env.P[state][action] = [
                    (self.p_nominal, *nominal),
                    (self.p_slip, *slip_1),
                    (self.p_slip, *slip_2),
                ]


class TestFrozenLakeProbabilities(unittest.TestCase):
    def setUp(self):
        self.p_nom = 0.8
        self.p_slip = 0.1
        base_env = gym.make("FrozenLake-v1", is_slippery=True)
        self.env = SlipperyProbabilityWrapper(
            base_env,
            p_nominal=self.p_nom,
            p_slip=self.p_slip,
        )

    def test_all_transitions_internal_matrix(self):
        inner_env = self.env.unwrapped
        p_matrix = inner_env.P
        n_states = inner_env.nrow * inner_env.ncol

        for state in range(n_states):
            for action in ACTIONS:
                transitions = p_matrix[state][action]
                self.assertEqual(
                    len(transitions),
                    3,
                    f"State {state}, action {action}: expected 3 transitions.",
                )

                probs = [t[0] for t in transitions]
                count_nom = sum(1 for p in probs if abs(p - self.p_nom) < EPS)
                count_slip = sum(1 for p in probs if abs(p - self.p_slip) < EPS)

                self.assertEqual(
                    count_nom,
                    1,
                    f"State {state}, action {action}: nominal probability not found once.",
                )
                self.assertEqual(
                    count_slip,
                    2,
                    f"State {state}, action {action}: slip probability not found twice.",
                )
                self.assertAlmostEqual(
                    sum(probs),
                    1.0,
                    places=7,
                    msg=f"State {state}, action {action}: probabilities do not sum to 1.",
                )

    def test_nominal_action_integrity(self):
        inner_env = self.env.unwrapped
        p_matrix = inner_env.P
        n_states = inner_env.nrow * inner_env.ncol

        for state in range(n_states):
            for action in ACTIONS:
                expected_next, _, _ = self.env._next_state_data(state, action)
                nominal_transitions = [
                    transition
                    for transition in p_matrix[state][action]
                    if abs(transition[0] - self.p_nom) < EPS
                ]
                self.assertEqual(
                    len(nominal_transitions),
                    1,
                    f"State {state}, action {action}: expected one nominal transition.",
                )
                actual_next = nominal_transitions[0][1]
                self.assertEqual(
                    actual_next,
                    expected_next,
                    f"State {state}, action {action}: nominal next-state mismatch.",
                )


if __name__ == "__main__":
    unittest.main()
