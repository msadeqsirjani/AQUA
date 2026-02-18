"""
DDPG Agent for HAQ (Wang et al., CVPR 2019).
Modernized from lib/rl/ddpg.py.

Changes from original:
  - Removed torch.autograd.Variable (deprecated)
  - Removed volatile=True, use torch.no_grad()
  - Clean replay buffer (no keras-rl dependency)
  - Ornstein-Uhlenbeck noise as specified in paper
  - Actor uses Sigmoid (not Tanh+clip) matching original code
  - Critic merges state+action in first hidden layer matching original

Architecture (Section 3.3):
  Actor:  Linear(9, 300) -> ReLU -> Linear(300, 300) -> ReLU -> Linear(300, 1) -> Sigmoid
  Critic: Linear(9, 300) + Linear(1, 300) -> ReLU -> Linear(300, 300) -> ReLU -> Linear(300, 1)
"""

import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from scipy import stats


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


# ---------------------------------------------------------------------------
# Actor and Critic Networks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """DDPG actor: state -> action in [0, 1]."""

    def __init__(self, state_dim, action_dim=1, hidden1=300, hidden2=300):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))
        return out


class Critic(nn.Module):
    """DDPG critic: (state, action) -> Q-value."""

    def __init__(self, state_dim, action_dim=1, hidden1=300, hidden2=300):
        super().__init__()
        self.fc_s = nn.Linear(state_dim, hidden1)
        self.fc_a = nn.Linear(action_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        out = self.fc_s(state) + self.fc_a(action)
        out = self.relu(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Simple replay buffer with uniform sampling."""

    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(
            state=np.array(state, dtype=np.float32),
            action=np.array(action, dtype=np.float32).reshape(-1),
            reward=float(reward),
            next_state=np.array(next_state, dtype=np.float32),
            done=float(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch]).reshape(-1, 1)
        next_states = np.array([t.next_state for t in batch])
        dones = np.array([t.done for t in batch]).reshape(-1, 1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck Noise
# ---------------------------------------------------------------------------

class OUNoise:
    """Ornstein-Uhlenbeck process for exploration.

    dx = theta * (mu - x) * dt + sigma * N(0,1) * sqrt(dt)
    With dt = 1 (discrete time steps).
    """

    def __init__(self, size=1, theta=0.9, sigma=0.5, mu=0.0):
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.size = size
        self.x = np.zeros(size)

    def reset(self):
        self.x = np.zeros(self.size)

    def sample(self):
        dx = self.theta * (self.mu - self.x) + self.sigma * np.random.randn(self.size)
        self.x += dx
        return self.x.copy()


# ---------------------------------------------------------------------------
# DDPG Agent
# ---------------------------------------------------------------------------

class DDPGAgent:
    """DDPG agent for HAQ mixed-precision search.

    Hyperparameters (matching paper Section 3.3):
      - Actor LR: 1e-4
      - Critic LR: 1e-3
      - Soft target update tau: 0.01
      - Discount gamma: 1.0 (episodic)
      - Replay buffer: 2000
      - Batch size: 64
      - Exploration: truncated normal with decaying sigma

    Args:
        state_dim: Dimension of state vector (9)
        action_dim: Dimension of action (1)
        device: torch device
        lr_actor: Actor learning rate
        lr_critic: Critic learning rate
        tau: Soft target update rate
        gamma: Discount factor
        buffer_size: Replay buffer capacity
        batch_size: Training batch size
        warmup_episodes: Episodes of random exploration before training
        init_delta: Initial noise sigma for truncated normal
        delta_decay: Decay rate per episode for noise
    """

    def __init__(self, state_dim, action_dim=1, device="cpu",
                 lr_actor=1e-4, lr_critic=1e-3, tau=0.01, gamma=1.0,
                 buffer_size=2000, batch_size=64, warmup_episodes=20,
                 init_delta=0.5, delta_decay=0.99):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_episodes = warmup_episodes
        self.init_delta = init_delta
        self.delta_decay = delta_decay

        # Networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)

        # Initialize targets with same weights
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)

        # Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Moving average for reward normalization
        self.moving_average = None
        self.moving_alpha = 0.5

        # Loss tracking
        self.value_loss = 0.0
        self.policy_loss = 0.0

    def select_action(self, state, episode):
        """Select action using actor + truncated normal noise.

        During warmup: random uniform [0, 1].
        After warmup: actor output + decaying truncated normal noise.
        """
        if episode < self.warmup_episodes:
            return np.random.uniform(0, 1, self.action_dim)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().squeeze(0)

        # Add truncated normal noise with decay
        delta = self.init_delta * (self.delta_decay ** (episode - self.warmup_episodes))
        noisy_action = stats.truncnorm.rvs(
            (0.0 - action) / max(delta, 1e-8),
            (1.0 - action) / max(delta, 1e-8),
            loc=action,
            scale=delta,
            size=self.action_dim,
        )
        return np.clip(noisy_action, 0.0, 1.0)

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def update_policy(self):
        """Sample from replay buffer and update actor/critic."""
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Reward normalization with moving average
        batch_mean = rewards.mean()
        if self.moving_average is None:
            self.moving_average = batch_mean
        else:
            self.moving_average += self.moving_alpha * (batch_mean - self.moving_average)
        rewards = rewards - self.moving_average

        # Convert to tensors
        s = torch.tensor(states, dtype=torch.float32, device=self.device)
        a = torch.tensor(actions, dtype=torch.float32, device=self.device)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        not_done = torch.tensor(1.0 - dones, dtype=torch.float32, device=self.device)

        # --- Critic update ---
        with torch.no_grad():
            next_actions = self.actor_target(s2)
            target_q = r + self.gamma * not_done * self.critic_target(s2, next_actions)

        current_q = self.critic(s, a)
        critic_loss = nn.functional.mse_loss(current_q, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # --- Actor update ---
        actor_loss = -self.critic(s, self.actor(s)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # --- Soft target update ---
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        self.value_loss = critic_loss.item()
        self.policy_loss = actor_loss.item()

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tp.data * (1.0 - self.tau) + sp.data * self.tau)

    def _hard_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(sp.data)

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
