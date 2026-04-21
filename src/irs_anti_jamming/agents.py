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
    """Fast Q-Learning [19] with visit-count-dependent learning rate boost."""

    def _alpha(self, state_id: int, action: int) -> float:
        visits = self.visits[state_id][action]
        # Boosted learning rate that decays with experience
        # With 30 actions, each action gets visited more frequently
        boost = 3.0 / (1.0 + 0.1 * visits)
        return min(1.0, self.cfg.alpha * (1.0 + boost))

    def end_episode(self) -> None:
        faster_decay = self.cfg.epsilon_decay ** 1.5
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * faster_decay)

    def set_eval_mode(self) -> None:
        self.epsilon = 0.0


class _ReplayBuffer:
    """Lightweight circular replay buffer for tabular experience replay."""
    __slots__ = ('capacity', 'buffer', 'pos', 'rng')

    def __init__(self, capacity: int = 256, rng: np.random.Generator | None = None):
        self.capacity = capacity
        self.buffer: list[tuple] = []
        self.pos = 0
        self.rng = rng or np.random.default_rng()

    def add(self, transition: tuple) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, n: int) -> list[tuple]:
        n = min(n, len(self.buffer))
        indices = self.rng.choice(len(self.buffer), size=n, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class FuzzyWoLFPHCAgent:
    """Enhanced Fuzzy WoLF-PHC agent — proposed method.

    Architecture per paper Equations 17-22 with key enhancements:
    - Q_l(s,a) table for each fuzzy state l, keyed by DISCRETE state s.
      This gives the agent the same state resolution as tabular Q-learning.
    - Fuzzy Q: FQ(s,a) = sum_l Q_l(s,a) * psi_l  (Eq.17)
    - Mixed policy pi_l(a) per fuzzy state l, updated via WoLF-PHC (Eq.14-16,20)

    Enhancements over baseline WoLF-PHC:
    1. Adaptive learning rate: visit-count-dependent alpha boost for faster convergence
    2. Adaptive Boltzmann softmax eval policy: temperature scales with Q-value spread
       for smart exploitation while maintaining anti-jammer unpredictability
    3. Experience replay: circular buffer of past transitions for faster Q-value
       convergence and more accurate value estimates
    4. Uniform mixing floor: minimum probability guarantee for all actions ensures
       the smart jammer can never fully predict the agent's behavior
    """

    def __init__(self, n_actions: int, n_fuzzy_states: int, cfg: RLConfig, seed: int = 0):
        self.n_actions = n_actions
        self.n_fuzzy_states = n_fuzzy_states
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        # Q_l(s,a) for each fuzzy state l, keyed by discrete state id
        # This is the key difference: Q-tables are per (fuzzy_state, discrete_state, action)
        self.q: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros((self.n_fuzzy_states, self.n_actions), dtype=float)
        )

        # pi_l(a) mixed policy per fuzzy state l (Eq.19-20)
        self.pi = np.full((n_fuzzy_states, n_actions), 1.0 / n_actions, dtype=float)
        # Average policy (Eq.16)
        self.pi_avg = self.pi.copy()
        # Visit counts C(l) per fuzzy state (Eq.16)
        self.count = np.ones(n_fuzzy_states, dtype=float)

        # Visit counts per (discrete_state, action) for adaptive learning rate
        self.visits = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))

        # Experience replay buffer for faster Q-value convergence
        self._replay = _ReplayBuffer(capacity=256, rng=self.rng)
        self._replay_batch = 4  # transitions replayed per step

        # Base Boltzmann temperature (adaptive scaling applied at eval time)
        self._eval_temperature = getattr(cfg, 'wolf_eval_temperature', 1.5)
        # Minimum uniform mixing ratio for unpredictability guarantee
        self._uniform_floor = 0.03

        self.epsilon = cfg.epsilon_start

    def _fuzzy_q(self, discrete_id: int, memberships: np.ndarray) -> np.ndarray:
        """FQ(s,a) = sum_l Q_l(s,a) * psi_l(s)  (Eq.17)"""
        q_table = self.q[discrete_id]  # (n_fuzzy_states, n_actions)
        return memberships @ q_table  # (n_actions,)

    def _softmax_probs(self, q_values: np.ndarray, temperature: float) -> np.ndarray:
        """Boltzmann softmax with temperature scaling for stochastic action selection."""
        q_scaled = q_values / max(temperature, 1e-6)
        q_scaled = q_scaled - q_scaled.max()  # numerical stability
        exp_q = np.exp(q_scaled)
        probs = exp_q / max(exp_q.sum(), 1e-12)
        probs = np.clip(probs, 1e-12, None)
        probs /= probs.sum()
        return probs

    def select_action(self, state: StateRepresentation) -> int:
        """Standard epsilon-greedy training; adaptive Boltzmann softmax eval.

        Training: epsilon-greedy over FQ for stable Q-value convergence.
        Eval: Adaptive Boltzmann softmax — temperature scales with Q-value spread
        so the agent exploits high-value actions when confident, and randomizes
        when Q-values are close (minimal cost to randomize). Uniform mixing floor
        ensures the smart jammer can never perfectly predict the agent's behavior.
        """
        fq = self._fuzzy_q(state.discrete_id, state.fuzzy_memberships)

        if self.epsilon > 0.0:
            # Training: standard epsilon-greedy over FQ
            if self.rng.random() < self.epsilon:
                return int(self.rng.integers(0, self.n_actions))
            return int(np.argmax(fq))
        else:
            # Eval: adaptive Boltzmann softmax for anti-jammer unpredictability.
            q_range = float(np.max(fq) - np.min(fq))
            if q_range < 1e-6:
                # All Q-values essentially equal → uniform random
                return int(self.rng.integers(0, self.n_actions))

            # Temperature adapts to Q-value spread:
            #   Large spread → lower temp → exploit the clear best action
            #   Small spread → higher temp → spread probability (cheap randomization)
            # This gives ~exp(3) ≈ 20x ratio between best/worst actions,
            # so top ~3 actions get ~60% of probability.
            adaptive_temp = max(0.15, q_range / 3.0)
            probs = self._softmax_probs(fq, adaptive_temp)

            # Mix with uniform distribution for minimum unpredictability guarantee.
            # Even with very skewed Q-values, every action has at least some chance.
            u = self._uniform_floor
            probs = (1.0 - u) * probs + u / self.n_actions
            probs /= probs.sum()

            return int(self.rng.choice(self.n_actions, p=probs))

    def _update_q_values(self, state: StateRepresentation, action: int, reward: float,
                         next_state: StateRepresentation, use_adaptive_alpha: bool = True) -> None:
        """Update Q-values via fuzzy Bellman equation (no WoLF policy step)."""
        psi = state.fuzzy_memberships
        psi_next = next_state.fuzzy_memberships
        sid = state.discrete_id
        sid_next = next_state.discrete_id

        # FQ(s',a') for Bellman target (Eq.22)
        fq_next = self._fuzzy_q(sid_next, psi_next)
        target = reward + self.cfg.gamma * float(np.max(fq_next))

        if use_adaptive_alpha:
            # Adaptive learning rate: boost for less-visited state-action pairs
            visit_count = self.visits[sid][action]
            alpha_boost = 3.0 / (1.0 + 0.1 * visit_count)
            alpha = min(1.0, self.cfg.alpha * (1.0 + alpha_boost))
            self.visits[sid][action] += 1.0
        else:
            # Fixed lower alpha for replay updates (avoid overwriting fresh data)
            alpha = self.cfg.alpha * 0.7

        q_table = self.q[sid]
        for ell in range(self.n_fuzzy_states):
            if psi[ell] <= 0.0:
                continue
            q_la = q_table[ell, action]
            q_table[ell, action] = (1.0 - alpha) * q_la + alpha * target

    def _update_wolf_policy(self, state: StateRepresentation, action: int) -> None:
        """WoLF-PHC mixed policy update (Eq.14-16, 20)."""
        psi = state.fuzzy_memberships
        sid = state.discrete_id
        q_table = self.q[sid]

        # Recompute FQ after Q update for policy improvement
        fq = self._fuzzy_q(sid, psi)
        best_action = int(np.argmax(fq))

        for ell in range(self.n_fuzzy_states):
            if psi[ell] <= 0.0:
                continue

            # WoLF: compare current vs average policy (Eq.15)
            ev_current = float(np.dot(self.pi[ell], q_table[ell]))
            ev_avg = float(np.dot(self.pi_avg[ell], q_table[ell]))
            xi = self.cfg.xi_win if ev_current > ev_avg else self.cfg.xi_loss

            # Policy update (Eq.20): delta = xi * psi_l
            delta = xi * psi[ell]
            delta_other = delta / max(1, self.n_actions - 1)

            for a in range(self.n_actions):
                if a == best_action:
                    self.pi[ell, a] += delta
                else:
                    self.pi[ell, a] -= delta_other

            # Project to valid probability distribution
            self.pi[ell, :] = np.clip(self.pi[ell, :], 1e-9, None)
            self.pi[ell, :] /= self.pi[ell, :].sum()

            # Update average policy (Eq.16)
            self.pi_avg[ell] += (self.pi[ell] - self.pi_avg[ell]) / self.count[ell]
            self.count[ell] += 1.0

    def update(self, state: StateRepresentation, action: int, reward: float, next_state: StateRepresentation) -> None:
        # Store transition for experience replay
        self._replay.add((state, action, reward, next_state))

        # Full update: Q-values + WoLF policy step
        self._update_q_values(state, action, reward, next_state, use_adaptive_alpha=True)
        self._update_wolf_policy(state, action)

        # Experience replay: replay past transitions (Q-values only, no policy step)
        # This improves Q-value accuracy without disturbing WoLF policy dynamics
        if len(self._replay) >= 16:
            for s, a, r, ns in self._replay.sample(self._replay_batch):
                self._update_q_values(s, a, r, ns, use_adaptive_alpha=False)

    def end_episode(self) -> None:
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)

    def set_eval_mode(self) -> None:
        self.epsilon = 0.0
