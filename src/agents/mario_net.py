"""Custom DDQN agent for Mario based on the PyTorch tutorial.

Reference: https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage


class MarioNet(nn.Module):
    """CNN for Double DQN.

    Architecture: (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output

    Has two identical networks:
    - online: actively trained
    - target: periodically synced from online (provides stable TD targets)
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = self._build_cnn(c, output_dim)
        self.target = self._build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Freeze target network
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def _build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )


class Mario:
    """DDQN Agent that acts, remembers, and learns.

    Hyperparameters from the PyTorch Mario tutorial.
    """

    def __init__(self, state_dim, action_dim, save_dir,
                 lr=0.00025,
                 gamma=0.9,
                 exploration_rate=1.0,
                 exploration_rate_decay=0.99999975,
                 exploration_rate_min=0.1,
                 batch_size=32,
                 buffer_size=100_000,
                 burnin=1e4,
                 learn_every=3,
                 sync_every=1e4,
                 save_every=5e5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Network
        self.net = MarioNet(self.state_dim, self.action_dim).float().to(self.device)

        # Exploration
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.exploration_rate_min = exploration_rate_min
        self.curr_step = 0

        # Memory
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(buffer_size, device=torch.device("cpu"))
        )
        self.batch_size = batch_size

        # Learning
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = torch.nn.SmoothL1Loss()

        # Timing
        self.burnin = int(burnin)
        self.learn_every = int(learn_every)
        self.sync_every = int(sync_every)
        self.save_every = int(save_every)

    def _to_channels_first(self, obs):
        """Convert (H, W, C) numpy array to (C, H, W) tensor."""
        arr = obs[0].__array__() if isinstance(obs, tuple) else obs.__array__()
        # (H, W, C) -> (C, H, W)
        if arr.ndim == 3 and arr.shape[2] < arr.shape[0]:
            arr = np.transpose(arr, (2, 0, 1))
        return arr.astype(np.float32) / 255.0

    def act(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state_t = self._to_channels_first(state)
            state_t = torch.tensor(state_t, device=self.device).unsqueeze(0)
            action_values = self.net(state_t, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # Decay exploration
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """Store experience in replay buffer."""
        state = torch.tensor(self._to_channels_first(state))
        next_state = torch.tensor(self._to_channels_first(next_state))
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        self.memory.add(TensorDict({
            "state": state,
            "next_state": next_state,
            "action": action,
            "reward": reward,
            "done": done,
        }, batch_size=[]))

    def recall(self):
        """Sample a batch from memory."""
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (
            batch.get(key) for key in ("state", "next_state", "action", "reward", "done")
        )
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        """Q_online(s, a)"""
        current_Q = self.net(state, model="online")[np.arange(0, self.batch_size), action]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """Double DQN target: r + gamma * Q_target(s', argmax_a Q_online(s', a))"""
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        """Backprop on the online network."""
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        """Copy online weights to target network."""
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        """Save checkpoint."""
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def load(self, path):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["model"])
        self.exploration_rate = checkpoint["exploration_rate"]
        print(f"Loaded model from {path}, exploration_rate={self.exploration_rate:.4f}")

    def learn(self):
        """Full learning step: sync target, save, sample, update."""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample and learn
        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)

        return td_est.mean().item(), loss


class MetricLogger:
    """Log and display training metrics."""

    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Current episode accumulators
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss is not None:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length > 0:
            self.ep_avg_losses.append(self.curr_ep_loss / self.curr_ep_loss_length)
            self.ep_avg_qs.append(self.curr_ep_q / self.curr_ep_loss_length)
        else:
            self.ep_avg_losses.append(0)
            self.ep_avg_qs.append(0)

        # Reset
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon:.3f} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q {mean_ep_q}"
        )
