from __future__ import annotations

from collections import defaultdict

import numpy as np

from .config import RLConfig
from .state import StateRepresentation


class TabularQAgent:
    def __init__(self, n_actions: int, cfg: RLConfig, seed: int = 0):
        self.n_actions = n_actions
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        self.q = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))
        self.visits = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))

        self.epsilon = cfg.epsilon_start

    def _alpha(self, state_id: int, action: int) -> float:
        return self.cfg.alpha

    def select_action(self, state: StateRepresentation) -> int:
        q_values = self.q[state.discrete_id]
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(q_values))

    def update(self, state: StateRepresentation, action: int, reward: float, next_state: StateRepresentation) -> None:
        q_s = self.q[state.discrete_id]
        q_next = self.q[next_state.discrete_id]
        alpha = self._alpha(state.discrete_id, action)
        target = reward + self.cfg.gamma * float(np.max(q_next))
        q_s[action] = (1.0 - alpha) * q_s[action] + alpha * target
        self.visits[state.discrete_id][action] += 1.0

    def end_episode(self) -> None:
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)

    def set_eval_mode(self) -> None:
        self.epsilon = 0.0


class FastQAgent(TabularQAgent):
    def _alpha(self, state_id: int, action: int) -> float:
        visits = self.visits[state_id][action]
        boost = 2.5 / (1.0 + 0.05 * visits)
        return min(1.0, self.cfg.alpha * (1.0 + boost))

    def end_episode(self) -> None:
        faster_decay = self.cfg.epsilon_decay ** 1.5
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * faster_decay)


class FuzzyWoLFPHCAgent:
    def __init__(self, n_actions: int, n_fuzzy_states: int, cfg: RLConfig, seed: int = 0):
        self.n_actions = n_actions
        self.n_fuzzy_states = n_fuzzy_states
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        self.q = np.zeros((n_fuzzy_states, n_actions), dtype=float)
        self.pi = np.full((n_fuzzy_states, n_actions), 1.0 / n_actions, dtype=float)
        self.pi_avg = self.pi.copy()
        self.count = np.ones(n_fuzzy_states, dtype=float)

        self.epsilon = cfg.epsilon_start

    def _fuzzy_q(self, memberships: np.ndarray) -> np.ndarray:
        return memberships @ self.q

    def _fuzzy_policy_value(self, memberships: np.ndarray) -> np.ndarray:
        return np.sum((self.pi * self.q) * memberships[:, None], axis=0)

    def _fuzzy_mixed_policy(self, memberships: np.ndarray) -> np.ndarray:
        pi_state = memberships @ self.pi
        pi_state = np.clip(pi_state, 1e-12, None)
        return pi_state / pi_state.sum()

    def _softmax(self, scores: np.ndarray, temperature: float = 0.25) -> np.ndarray:
        t = max(1e-3, float(temperature))
        shifted = (scores - np.max(scores)) / t
        exp_scores = np.exp(shifted)
        exp_scores = np.clip(exp_scores, 1e-12, None)
        return exp_scores / exp_scores.sum()

    def select_action(self, state: StateRepresentation) -> int:
        policy_scores = self._fuzzy_policy_value(state.fuzzy_memberships)
        action_probs = self._softmax(policy_scores, temperature=0.20)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(self.rng.choice(self.n_actions, p=action_probs))

    def update(self, state: StateRepresentation, action: int, reward: float, next_state: StateRepresentation) -> None:
        psi = state.fuzzy_memberships
        psi_next = next_state.fuzzy_memberships

        fq = self._fuzzy_q(psi)
        fq_next = self._fuzzy_q(psi_next)

        target = reward + self.cfg.gamma * float(np.max(fq_next))
        td = target - float(fq[action])

        self.q[:, action] += self.cfg.alpha * psi * td

        best_action = int(np.argmax(fq))
        for l in range(self.n_fuzzy_states):
            if psi[l] <= 0.0:
                continue

            ev_current = float(np.dot(self.pi[l], self.q[l]))
            ev_avg = float(np.dot(self.pi_avg[l], self.q[l]))
            xi = self.cfg.xi_win if ev_current > ev_avg else self.cfg.xi_loss

            inc = xi * psi[l] / self.n_fuzzy_states
            dec = inc / max(1, self.n_actions - 1)

            self.pi[l, :] -= dec
            self.pi[l, best_action] += inc + dec
            self.pi[l, :] = np.clip(self.pi[l, :], 1e-9, None)
            self.pi[l, :] /= self.pi[l, :].sum()

            self.pi_avg[l] += (self.pi[l] - self.pi_avg[l]) / self.count[l]
            self.count[l] += 1.0

    def end_episode(self) -> None:
        # WoLF-PHC's softmax policy provides exploration; needs less epsilon
        wolfphc_eps_end = self.cfg.epsilon_end * 0.2
        self.epsilon = max(wolfphc_eps_end, self.epsilon * self.cfg.epsilon_decay)

    def set_eval_mode(self) -> None:
        self.epsilon = 0.0
