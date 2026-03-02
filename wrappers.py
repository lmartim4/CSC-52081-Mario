"""
Reward shaping wrappers for MountainCarContinuous-v0.

Four strategies compared:
  1. DenseHeightWrapper       — height-gain bonus (classic dense shaping)
  2. PBRSEnergyWrapper        — Potential-Based Reward Shaping (Ng et al. 1999)
  3. CuriosityRNDWrapper      — Random Network Distillation (Burda et al. 2018)
  4. ProgressiveDecayWrapper  — PBRS with annealing coef (original experiment)
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _height(position: float) -> float:
    """True height profile of MountainCarContinuous (same as Gymnasium source)."""
    return np.sin(3.0 * position)


def _phi(obs: np.ndarray, energy_scale: float = 1.0) -> float:
    """Mechanical-energy potential: kinetic + potential energy."""
    position, velocity = float(obs[0]), float(obs[1])
    kinetic = 0.5 * velocity ** 2
    potential = _height(position)
    return energy_scale * (kinetic + potential)


# ---------------------------------------------------------------------------
# 1. Dense height-based reward
# ---------------------------------------------------------------------------

class DenseHeightWrapper(gym.RewardWrapper):
    """
    Adds a bonus proportional to height *gain* each step.

    F(s, s') = alpha * max(0, h(s') - h(s))

    This keeps the agent incentivised to move upward even when no goal
    is reached, turning the sparse problem into a dense one.
    """

    def __init__(self, env: gym.Env, alpha: float = 0.5):
        super().__init__(env)
        self.alpha = alpha
        self._prev_height: float = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_height = _height(float(obs[0]))
        return obs, info

    def reward(self, reward: float) -> float:
        # Note: at this point env has already stepped, so we need current obs.
        # We override step() instead to have access to both obs.
        return reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_height = _height(float(obs[0]))
        bonus = self.alpha * max(0.0, current_height - self._prev_height)
        self._prev_height = current_height
        return obs, float(reward) + bonus, terminated, truncated, info


# ---------------------------------------------------------------------------
# 2. Potential-Based Reward Shaping (PBRS)
# ---------------------------------------------------------------------------

class PBRSEnergyWrapper(gym.Wrapper):
    """
    Potential-Based Reward Shaping using mechanical energy as potential.

        F(s, s') = gamma * Phi(s') - Phi(s)
        Phi(s)   = 0.5*v^2 + sin(3*x)   (kinetic + potential energy)

    Theoretical guarantee (Ng et al. 1999): PBRS never changes the set of
    optimal policies — it only accelerates convergence.
    """

    def __init__(self, env: gym.Env, gamma: float = 0.99, energy_scale: float = 1.0):
        super().__init__(env)
        self.gamma = gamma
        self.energy_scale = energy_scale
        self._prev_phi: float = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = _phi(obs, self.energy_scale)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_phi = _phi(obs, self.energy_scale)
        shaping = self.gamma * current_phi - self._prev_phi
        self._prev_phi = current_phi
        return obs, float(reward) + shaping, terminated, truncated, info


# ---------------------------------------------------------------------------
# 3. Curiosity — Random Network Distillation (RND)
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CuriosityRNDWrapper(gym.Wrapper):
    """
    Intrinsic curiosity via Random Network Distillation (Burda et al. 2018).

    A fixed random *target* network maps observations to a feature space.
    A *predictor* network is trained online to match the target.
    The prediction error is the intrinsic reward — novel states are harder
    to predict, giving higher bonuses and encouraging exploration.

        r_intrinsic = || target(s') - predictor(s') ||^2
    """

    def __init__(
        self,
        env: gym.Env,
        feature_dim: int = 64,
        lr: float = 1e-3,
        intrinsic_coef: float = 0.1,
    ):
        super().__init__(env)
        obs_dim = env.observation_space.shape[0]

        self.target = _MLP(obs_dim, 64, feature_dim)
        self.predictor = _MLP(obs_dim, 64, feature_dim)
        # Target is fixed — disable gradients
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.intrinsic_coef = intrinsic_coef

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs_t = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():
            target_feat = self.target(obs_t)

        pred_feat = self.predictor(obs_t)
        loss = (target_feat - pred_feat).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        intrinsic = loss.item()
        return obs, float(reward) + self.intrinsic_coef * intrinsic, terminated, truncated, info


# ---------------------------------------------------------------------------
# 4. Progressive Decay (original experiment)
# ---------------------------------------------------------------------------

class ProgressiveDecayWrapper(gym.Wrapper):
    """
    **Original experiment**: PBRS with a linearly-decaying shaping coefficient.

    Motivation:
      - Pure PBRS accelerates learning but the agent may become *dependent*
        on the shaping signal and not fully optimise the true reward.
      - By annealing the coefficient from 1 → 0 over training, we get the
        benefits of guided exploration early on, while guaranteeing that the
        final policy is evaluated against the sparse reward alone.

    Shaping bonus:
        coef(t) = max(0,  1 - t / anneal_steps)
        F(s,s') = coef(t) * [gamma * Phi(s') - Phi(s)]

    This is related to curriculum learning and reward annealing in the
    literature (Bengio et al. 2009; Mnih et al. 2015 reward clipping).
    """

    def __init__(
        self,
        env: gym.Env,
        anneal_steps: int = 80_000,
        gamma: float = 0.99,
        energy_scale: float = 1.0,
    ):
        super().__init__(env)
        self.anneal_steps = anneal_steps
        self.gamma = gamma
        self.energy_scale = energy_scale
        self._total_steps: int = 0
        self._prev_phi: float = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = _phi(obs, self.energy_scale)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_phi = _phi(obs, self.energy_scale)

        coef = max(0.0, 1.0 - self._total_steps / self.anneal_steps)
        shaping = coef * (self.gamma * current_phi - self._prev_phi)

        self._total_steps += 1
        self._prev_phi = current_phi
        return obs, float(reward) + shaping, terminated, truncated, info

    @property
    def current_coef(self) -> float:
        return max(0.0, 1.0 - self._total_steps / self.anneal_steps)
