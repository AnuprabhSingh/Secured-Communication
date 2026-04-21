"""DQN agent for THz anti-jamming system.

Supports two backends:
  - PyTorch (preferred, if available): standard DQN with target network
  - NumPy fallback: simple 2-layer MLP with manual backprop

The agent interface matches the tabular agents: select_action() and update().
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Agent interface matching existing tabular agents
# ---------------------------------------------------------------------------

class DQNAgent:
    """DQN agent for RL-based anti-jamming.

    Uses experience replay, target network, and epsilon-greedy exploration.
    Compatible with both tabular state IDs (used via embedding) and
    continuous state features.

    Args:
        n_actions: size of discrete action space (30)
        state_dim: dimension of continuous state vector (3)
        rl_cfg: THzRLConfig or compatible object with DQN hyperparams
        seed: random seed
    """

    def __init__(self, n_actions: int, state_dim: int = 3,
                 rl_cfg=None, seed: int = 0):
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)

        # Hyperparams (use defaults if rl_cfg not provided)
        self.gamma = getattr(rl_cfg, "gamma", 0.9)
        self.epsilon = getattr(rl_cfg, "epsilon_start", 1.0)
        self.epsilon_end = getattr(rl_cfg, "epsilon_end", 0.05)
        self.epsilon_decay = getattr(rl_cfg, "epsilon_decay", 0.995)
        self.hidden1 = getattr(rl_cfg, "dqn_hidden1", 128)
        self.hidden2 = getattr(rl_cfg, "dqn_hidden2", 64)
        self.replay_size = getattr(rl_cfg, "dqn_replay_size", 10_000)
        self.batch_size = getattr(rl_cfg, "dqn_batch_size", 64)
        self.tau = getattr(rl_cfg, "dqn_target_tau", 0.005)
        self.lr = getattr(rl_cfg, "dqn_lr", 1e-3)

        # Experience replay
        self.replay_buffer: deque = deque(maxlen=self.replay_size)

        if HAS_TORCH:
            self._init_torch()
        else:
            self._init_numpy()

        self.steps = 0

    # -------------------------------------------------------------------
    # PyTorch backend
    # -------------------------------------------------------------------

    def _init_torch(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        class QNet(nn.Module):
            def __init__(inner, s_dim, h1, h2, n_act):
                super().__init__()
                inner.net = nn.Sequential(
                    nn.Linear(s_dim, h1),
                    nn.ReLU(),
                    nn.Linear(h1, h2),
                    nn.ReLU(),
                    nn.Linear(h2, n_act),
                )

            def forward(inner, x):
                return inner.net(x)

        self.q_net = QNet(self.state_dim, self.hidden1, self.hidden2, self.n_actions).to(self.device)
        self.target_net = QNet(self.state_dim, self.hidden1, self.hidden2, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.backend = "torch"

    # -------------------------------------------------------------------
    # NumPy fallback backend
    # -------------------------------------------------------------------

    def _init_numpy(self):
        """Simple 2-layer MLP: state_dim -> h1 -> h2 -> n_actions."""
        self.backend = "numpy"

        def _init_weights(fan_in, fan_out, rng):
            scale = math.sqrt(2.0 / fan_in)
            W = rng.standard_normal((fan_in, fan_out)) * scale
            b = np.zeros(fan_out)
            return W.astype(np.float64), b.astype(np.float64)

        self.W1, self.b1 = _init_weights(self.state_dim, self.hidden1, self.rng)
        self.W2, self.b2 = _init_weights(self.hidden1, self.hidden2, self.rng)
        self.W3, self.b3 = _init_weights(self.hidden2, self.n_actions, self.rng)

        # Target copies
        self.W1_t, self.b1_t = self.W1.copy(), self.b1.copy()
        self.W2_t, self.b2_t = self.W2.copy(), self.b2.copy()
        self.W3_t, self.b3_t = self.W3.copy(), self.b3.copy()

    def _np_forward(self, state, use_target=False):
        """Forward pass through numpy MLP."""
        if use_target:
            W1, b1, W2, b2, W3, b3 = self.W1_t, self.b1_t, self.W2_t, self.b2_t, self.W3_t, self.b3_t
        else:
            W1, b1, W2, b2, W3, b3 = self.W1, self.b1, self.W2, self.b2, self.W3, self.b3

        z1 = state @ W1 + b1
        h1 = np.maximum(z1, 0)  # ReLU
        z2 = h1 @ W2 + b2
        h2 = np.maximum(z2, 0)
        q = h2 @ W3 + b3
        return q, (z1, h1, z2, h2)

    def _np_backward(self, state, action, target_q):
        """One-step gradient update for a single transition."""
        q_all, (z1, h1, z2, h2) = self._np_forward(state, use_target=False)
        loss_grad = np.zeros_like(q_all)
        loss_grad[action] = 2.0 * (q_all[action] - target_q)

        # Layer 3
        dW3 = np.outer(h2, loss_grad)
        db3 = loss_grad
        dh2 = loss_grad @ self.W3.T

        # ReLU 2
        dz2 = dh2 * (z2 > 0).astype(float)
        dW2 = np.outer(h1, dz2)
        db2 = dz2
        dh1 = dz2 @ self.W2.T

        # ReLU 1
        dz1 = dh1 * (z1 > 0).astype(float)
        dW1 = np.outer(state, dz1)
        db1 = dz1

        # SGD update
        for W, dW, b, db in [
            (self.W1, dW1, self.b1, db1),
            (self.W2, dW2, self.b2, db2),
            (self.W3, dW3, self.b3, db3),
        ]:
            W -= self.lr * np.clip(dW, -1.0, 1.0)
            b -= self.lr * np.clip(db, -1.0, 1.0)

    def _np_soft_update(self):
        """Soft update target network."""
        for src, tgt in [
            (self.W1, self.W1_t), (self.b1, self.b1_t),
            (self.W2, self.W2_t), (self.b2, self.b2_t),
            (self.W3, self.W3_t), (self.b3, self.b3_t),
        ]:
            tgt[:] = self.tau * src + (1.0 - self.tau) * tgt

    # -------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------

    def _state_to_array(self, state) -> np.ndarray:
        """Convert state (features array or StateRepresentation) to numpy array."""
        if hasattr(state, "features"):
            return np.asarray(state.features, dtype=np.float64)
        return np.asarray(state, dtype=np.float64).ravel()[:self.state_dim]

    def select_action(self, state, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))

        s = self._state_to_array(state)

        if self.backend == "torch":
            with torch.no_grad():
                s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_vals = self.q_net(s_t).squeeze(0).cpu().numpy()
        else:
            q_vals, _ = self._np_forward(s, use_target=False)

        return int(np.argmax(q_vals))

    def update(self, state, action: int, reward: float, next_state, done: bool = False):
        """Store transition and train on mini-batch."""
        s = self._state_to_array(state)
        s_next = self._state_to_array(next_state)
        self.replay_buffer.append((s, action, reward, s_next, done))
        self.steps += 1

        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample mini-batch
        indices = self.rng.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]

        if self.backend == "torch":
            self._torch_train(batch)
        else:
            self._np_train(batch)

    def _torch_train(self, batch):
        states = torch.tensor(np.array([t[0] for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([t[3] for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t[4] for t in batch], dtype=torch.float32, device=self.device)

        q_vals = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.functional.mse_loss(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target
        with torch.no_grad():
            for p, p_t in zip(self.q_net.parameters(), self.target_net.parameters()):
                p_t.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_t.data)

    def _np_train(self, batch):
        for s, a, r, s_next, done in batch:
            q_next, _ = self._np_forward(s_next, use_target=True)
            target = r + (0.0 if done else self.gamma * np.max(q_next))
            self._np_backward(s, a, target)
        self._np_soft_update()

    def decay_epsilon(self):
        """Call at end of each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
