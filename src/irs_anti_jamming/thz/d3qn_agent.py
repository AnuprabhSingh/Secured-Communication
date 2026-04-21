"""Dueling Double DQN with Prioritized Experience Replay (D3QN-PER).

Proposed method for IRS-aided THz anti-jamming secure communications.

Enhancements over vanilla DQN:
  1. Double DQN: uses online net for action selection, target net for
     value estimation → eliminates Q-value overestimation bias.
     [van Hasselt et al., "Deep RL with Double Q-learning", AAAI 2016]

  2. Dueling Architecture: separates state-value V(s) from advantage A(s,a)
     → better generalisation across actions, faster convergence.
     [Wang et al., "Dueling Network Architectures for Deep RL", ICML 2016]

  3. Prioritized Experience Replay (PER): samples transitions with high
     TD-error more frequently → efficient use of experience.
     [Schaul et al., "Prioritized Experience Replay", ICLR 2016]

  4. Noisy Networks (optional): parametric noise in linear layers for
     state-dependent exploration → no epsilon scheduling needed.
     [Fortunato et al., "Noisy Networks for Exploration", ICLR 2018]

The agent interface matches the tabular agents: select_action() and update().
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Prioritized Experience Replay (Sum-Tree implementation)
# ---------------------------------------------------------------------------

class SumTree:
    """Binary sum-tree for O(log n) priority-based sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_ptr = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    @property
    def total_priority(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data):
        tree_idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(tree_idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """PER buffer with proportional prioritisation."""

    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4,
                 beta_frames: int = 100_000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self.max_priority = 1.0
        self.epsilon = 1e-6

    @property
    def beta(self) -> float:
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def add(self, transition):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int):
        self.frame += 1
        indices = []
        priorities = []
        batch = []

        total = self.tree.total_priority
        segment = total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, prio, data = self.tree.get(s)
            if data is None:
                # Fallback: resample
                s = np.random.uniform(0, total)
                idx, prio, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(prio)
            batch.append(data)

        priorities = np.array(priorities, dtype=np.float64) + self.epsilon
        probs = priorities / self.tree.total_priority
        N = self.tree.size

        # Importance-sampling weights
        weights = (N * probs) ** (-self.beta)
        weights = weights / weights.max()

        return batch, np.array(indices), weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.size


# ---------------------------------------------------------------------------
# Dueling Network Architecture
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class NoisyLinear(nn.Module):
        """Factorised Gaussian noisy linear layer (Fortunato et al., 2018)."""

        def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

            self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
            self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
            self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
            self.register_buffer("bias_epsilon", torch.empty(out_features))

            self.sigma_init = sigma_init
            self._reset_parameters()
            self.reset_noise()

        def _reset_parameters(self):
            bound = 1.0 / math.sqrt(self.in_features)
            self.weight_mu.data.uniform_(-bound, bound)
            self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
            self.bias_mu.data.uniform_(-bound, bound)
            self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

        @staticmethod
        def _scale_noise(size: int) -> torch.Tensor:
            x = torch.randn(size)
            return x.sign() * x.abs().sqrt()

        def reset_noise(self):
            eps_in = self._scale_noise(self.in_features)
            eps_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(eps_out.outer(eps_in))
            self.bias_epsilon.copy_(eps_out)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.training:
                weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
                bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                weight = self.weight_mu
                bias = self.bias_mu
            return F.linear(x, weight, bias)


    class DuelingQNetwork(nn.Module):
        """Dueling architecture: shared feature layer → separate V(s) and A(s,a) streams.

        Q(s,a) = V(s) + A(s,a) - mean(A(s,·))  (Wang et al., 2016, Eq. 9)
        """

        def __init__(self, state_dim: int, n_actions: int,
                     hidden1: int = 256, hidden2: int = 128,
                     use_noisy: bool = True):
            super().__init__()

            Linear = NoisyLinear if use_noisy else nn.Linear

            # Shared feature extraction
            self.feature = nn.Sequential(
                nn.Linear(state_dim, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
            )

            # Value stream V(s)
            self.value_stream = nn.Sequential(
                Linear(hidden2, hidden2 // 2),
                nn.ReLU(),
                Linear(hidden2 // 2, 1),
            )

            # Advantage stream A(s,a)
            self.advantage_stream = nn.Sequential(
                Linear(hidden2, hidden2 // 2),
                nn.ReLU(),
                Linear(hidden2 // 2, n_actions),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.feature(x)
            value = self.value_stream(features)         # (batch, 1)
            advantage = self.advantage_stream(features)  # (batch, n_actions)
            # Q = V + A - mean(A)  → ensures identifiability
            q = value + advantage - advantage.mean(dim=-1, keepdim=True)
            return q

        def reset_noise(self):
            """Reset noise in all NoisyLinear layers."""
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


# ---------------------------------------------------------------------------
# D3QN-PER Agent
# ---------------------------------------------------------------------------

class D3QNAgent:
    """Dueling Double DQN with Prioritized Experience Replay.

    Proposed method for IRS-aided THz secure communication against smart jamming.

    Key advantages over tabular RL and vanilla DQN:
      - Handles continuous state spaces without discretisation loss
      - Double Q-learning eliminates overestimation bias
      - Dueling architecture improves value estimation efficiency
      - PER focuses learning on informative transitions
      - Noisy networks provide state-dependent exploration

    Args:
        n_actions: size of discrete action space
        state_dim: dimension of continuous state vector (3)
        rl_cfg: THzRLConfig or compatible object with hyperparams
        seed: random seed
    """

    def __init__(self, n_actions: int, state_dim: int = 3,
                 rl_cfg=None, seed: int = 0):
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)

        # Hyperparams — tuned for stable convergence in IRS anti-jamming
        self.gamma = getattr(rl_cfg, "gamma", 0.9)
        self.epsilon = getattr(rl_cfg, "epsilon_start", 1.0)
        self.epsilon_end = getattr(rl_cfg, "epsilon_end", 0.01)
        self.epsilon_decay = getattr(rl_cfg, "epsilon_decay", 0.995)
        self.hidden1 = getattr(rl_cfg, "dqn_hidden1", 256)
        self.hidden2 = getattr(rl_cfg, "dqn_hidden2", 128)
        self.replay_size = getattr(rl_cfg, "dqn_replay_size", 50_000)
        self.batch_size = getattr(rl_cfg, "dqn_batch_size", 128)
        self.tau = getattr(rl_cfg, "dqn_target_tau", 0.002)
        self.lr = getattr(rl_cfg, "dqn_lr", 5e-4)
        self.warmup_steps = getattr(rl_cfg, "dqn_warmup_steps", 256)
        self.train_freq = getattr(rl_cfg, "dqn_train_freq", 2)  # train every N steps

        # PER parameters
        self.per_alpha = getattr(rl_cfg, "per_alpha", 0.6)
        self.per_beta_start = getattr(rl_cfg, "per_beta_start", 0.4)

        # Use noisy nets for exploration
        self.use_noisy = getattr(rl_cfg, "use_noisy_nets", True)

        self.steps = 0

        if HAS_TORCH:
            self._init_torch()
        else:
            self._init_numpy_fallback()

    # -------------------------------------------------------------------
    # PyTorch backend (primary)
    # -------------------------------------------------------------------

    def _init_torch(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.q_net = DuelingQNetwork(
            self.state_dim, self.n_actions,
            self.hidden1, self.hidden2,
            use_noisy=self.use_noisy
        ).to(self.device)

        self.target_net = DuelingQNetwork(
            self.state_dim, self.n_actions,
            self.hidden1, self.hidden2,
            use_noisy=self.use_noisy
        ).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr, eps=1e-8)
        # Cosine-annealing LR schedule for stable convergence
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20_000, eta_min=self.lr * 0.1
        )

        # Prioritized Experience Replay
        total_steps = getattr(self, '_total_training_steps', 100_000)
        self.replay_buffer = PrioritizedReplayBuffer(
            self.replay_size,
            alpha=self.per_alpha,
            beta_start=self.per_beta_start,
            beta_frames=total_steps,
        )

        self.backend = "torch"

    # -------------------------------------------------------------------
    # NumPy fallback (for environments without PyTorch)
    # -------------------------------------------------------------------

    def _init_numpy_fallback(self):
        """Simplified D3QN with numpy: Double DQN + Dueling (no PER for simplicity)."""
        self.backend = "numpy"
        rng = self.rng

        def _init_weights(fan_in, fan_out):
            scale = math.sqrt(2.0 / fan_in)
            W = rng.standard_normal((fan_in, fan_out)) * scale
            b = np.zeros(fan_out)
            return W.astype(np.float64), b.astype(np.float64)

        # Shared features: state_dim → h1 → h2
        self.W1, self.b1 = _init_weights(self.state_dim, self.hidden1)
        self.W2, self.b2 = _init_weights(self.hidden1, self.hidden2)

        # Value stream: h2 → h2//2 → 1
        self.Wv1, self.bv1 = _init_weights(self.hidden2, self.hidden2 // 2)
        self.Wv2, self.bv2 = _init_weights(self.hidden2 // 2, 1)

        # Advantage stream: h2 → h2//2 → n_actions
        self.Wa1, self.ba1 = _init_weights(self.hidden2, self.hidden2 // 2)
        self.Wa2, self.ba2 = _init_weights(self.hidden2 // 2, self.n_actions)

        # Target copies
        self._copy_to_target()

        # Simple replay for numpy
        from collections import deque
        self.replay_buffer = deque(maxlen=self.replay_size)

    def _copy_to_target(self):
        self.W1_t, self.b1_t = self.W1.copy(), self.b1.copy()
        self.W2_t, self.b2_t = self.W2.copy(), self.b2.copy()
        self.Wv1_t, self.bv1_t = self.Wv1.copy(), self.bv1.copy()
        self.Wv2_t, self.bv2_t = self.Wv2.copy(), self.bv2.copy()
        self.Wa1_t, self.ba1_t = self.Wa1.copy(), self.ba1.copy()
        self.Wa2_t, self.ba2_t = self.Wa2.copy(), self.ba2.copy()

    def _np_forward(self, state, use_target=False):
        if use_target:
            W1, b1 = self.W1_t, self.b1_t
            W2, b2 = self.W2_t, self.b2_t
            Wv1, bv1, Wv2, bv2 = self.Wv1_t, self.bv1_t, self.Wv2_t, self.bv2_t
            Wa1, ba1, Wa2, ba2 = self.Wa1_t, self.ba1_t, self.Wa2_t, self.ba2_t
        else:
            W1, b1 = self.W1, self.b1
            W2, b2 = self.W2, self.b2
            Wv1, bv1, Wv2, bv2 = self.Wv1, self.bv1, self.Wv2, self.bv2
            Wa1, ba1, Wa2, ba2 = self.Wa1, self.ba1, self.Wa2, self.ba2

        z1 = state @ W1 + b1
        h1 = np.maximum(z1, 0)
        z2 = h1 @ W2 + b2
        h2 = np.maximum(z2, 0)

        # Value stream
        zv1 = h2 @ Wv1 + bv1
        hv1 = np.maximum(zv1, 0)
        value = hv1 @ Wv2 + bv2  # scalar

        # Advantage stream
        za1 = h2 @ Wa1 + ba1
        ha1 = np.maximum(za1, 0)
        advantage = ha1 @ Wa2 + ba2  # (n_actions,)

        # Q = V + A - mean(A)
        q = value + advantage - np.mean(advantage)
        return q, (z1, h1, z2, h2, zv1, hv1, za1, ha1)

    def _np_backward(self, state, action, target_q):
        q_all, cache = self._np_forward(state, use_target=False)
        z1, h1, z2, h2, zv1, hv1, za1, ha1 = cache

        # dL/dQ for the chosen action
        loss_grad_q = np.zeros(self.n_actions)
        loss_grad_q[action] = 2.0 * (q_all[action] - target_q)

        # Backprop through Q = V + A - mean(A)
        # dQ/dA_i = 1 - 1/n for i=action, -1/n otherwise
        dA = loss_grad_q.copy()
        dA -= np.mean(loss_grad_q)
        dV = np.sum(loss_grad_q)

        # Advantage stream backward
        dWa2 = np.outer(ha1, dA)
        dba2 = dA
        dha1 = dA @ self.Wa2.T
        dza1 = dha1 * (za1 > 0).astype(float)
        dWa1 = np.outer(h2, dza1)
        dba1 = dza1

        # Value stream backward
        dVscalar = np.array([dV])
        dWv2 = np.outer(hv1, dVscalar)
        dbv2 = dVscalar
        dhv1 = dVscalar @ self.Wv2.T
        dzv1 = dhv1 * (zv1 > 0).astype(float)
        dWv1 = np.outer(h2, dzv1)
        dbv1 = dzv1

        # Shared feature backward
        dh2 = dza1 @ self.Wa1.T + dzv1 @ self.Wv1.T
        dz2 = dh2 * (z2 > 0).astype(float)
        dW2 = np.outer(h1, dz2)
        db2 = dz2
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (z1 > 0).astype(float)
        dW1 = np.outer(state, dz1)
        db1 = dz1

        # SGD update with gradient clipping
        for W, dW, b, db in [
            (self.W1, dW1, self.b1, db1),
            (self.W2, dW2, self.b2, db2),
            (self.Wv1, dWv1, self.bv1, dbv1),
            (self.Wv2, dWv2, self.bv2, dbv2),
            (self.Wa1, dWa1, self.ba1, dba1),
            (self.Wa2, dWa2, self.ba2, dba2),
        ]:
            W -= self.lr * np.clip(dW, -1.0, 1.0)
            b -= self.lr * np.clip(db, -1.0, 1.0)

    def _np_soft_update(self):
        for src, tgt in [
            (self.W1, self.W1_t), (self.b1, self.b1_t),
            (self.W2, self.W2_t), (self.b2, self.b2_t),
            (self.Wv1, self.Wv1_t), (self.bv1, self.bv1_t),
            (self.Wv2, self.Wv2_t), (self.bv2, self.bv2_t),
            (self.Wa1, self.Wa1_t), (self.ba1, self.ba1_t),
            (self.Wa2, self.Wa2_t), (self.ba2, self.ba2_t),
        ]:
            tgt[:] = self.tau * src + (1.0 - self.tau) * tgt

    # -------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------

    def _state_to_array(self, state) -> np.ndarray:
        if hasattr(state, "features"):
            return np.asarray(state.features, dtype=np.float64)
        return np.asarray(state, dtype=np.float64).ravel()[:self.state_dim]

    def select_action(self, state, training: bool = True) -> int:
        """Action selection.

        With noisy nets: greedy w.r.t. noisy Q-values (exploration is implicit).
        Without noisy nets: epsilon-greedy.
        """
        if training and not self.use_noisy:
            if self.rng.random() < self.epsilon:
                return int(self.rng.integers(0, self.n_actions))

        s = self._state_to_array(state)

        if self.backend == "torch":
            with torch.no_grad():
                s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
                if training and self.use_noisy:
                    self.q_net.reset_noise()
                q_vals = self.q_net(s_t).squeeze(0).cpu().numpy()
        else:
            q_vals, _ = self._np_forward(s, use_target=False)

        return int(np.argmax(q_vals))

    def update(self, state, action: int, reward: float, next_state, done: bool = False):
        """Store transition and train on prioritised mini-batch."""
        s = self._state_to_array(state)
        s_next = self._state_to_array(next_state)

        if self.backend == "torch":
            self.replay_buffer.add((s, action, reward, s_next, done))
        else:
            self.replay_buffer.append((s, action, reward, s_next, done))

        self.steps += 1

        buf_len = len(self.replay_buffer)
        if buf_len < self.warmup_steps:
            return

        # Train every N steps to stabilise learning
        if self.steps % self.train_freq != 0:
            return

        if self.backend == "torch":
            self._torch_train()
        else:
            self._np_train()

    def _torch_train(self):
        """Double DQN training with PER and dueling network."""
        batch, tree_indices, is_weights = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(np.array([t[0] for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([t[3] for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t[4] for t in batch], dtype=torch.float32, device=self.device)
        weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device)

        # Reset noise for both networks
        if self.use_noisy:
            self.q_net.reset_noise()
            self.target_net.reset_noise()

        # Current Q-values
        q_vals = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # --- Double DQN: action selection with online net, evaluation with target ---
            next_q_online = self.q_net(next_states)
            best_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            target = rewards + self.gamma * next_q_target * (1.0 - dones)

        # TD errors for priority updates
        td_errors = (q_vals - target).detach().cpu().numpy()

        # Weighted Huber loss (more robust than MSE)
        element_wise_loss = F.smooth_l1_loss(q_vals, target, reduction="none")
        loss = (weights * element_wise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        self.lr_scheduler.step()

        # Update priorities in PER
        self.replay_buffer.update_priorities(tree_indices, td_errors)

        # Soft update target network
        with torch.no_grad():
            for p, p_t in zip(self.q_net.parameters(), self.target_net.parameters()):
                p_t.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_t.data)

    def _np_train(self):
        """NumPy fallback: Double DQN with dueling architecture."""
        indices = self.rng.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]

        for s, a, r, s_next, done in batch:
            # Double DQN: select action with online net, eval with target
            q_online_next, _ = self._np_forward(s_next, use_target=False)
            best_action = int(np.argmax(q_online_next))
            q_target_next, _ = self._np_forward(s_next, use_target=True)

            target = r + (0.0 if done else self.gamma * q_target_next[best_action])
            self._np_backward(s, a, target)

        self._np_soft_update()

    def decay_epsilon(self):
        """Call at end of each episode (only used when noisy nets disabled)."""
        if not self.use_noisy:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def set_eval_mode(self):
        """Switch to evaluation mode (no exploration noise)."""
        self.epsilon = 0.0
        if self.backend == "torch":
            self.q_net.eval()

    def end_episode(self):
        """Alias for decay_epsilon, for interface compatibility."""
        self.decay_epsilon()
