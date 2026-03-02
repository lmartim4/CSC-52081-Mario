"""
Experiment runner for reward shaping comparison on MountainCarContinuous-v0.

Uses Stable-Baselines3 PPO with a shared hyperparameter set so that
differences in training curves are attributable to the reward signal alone.
"""

import pickle
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


# ---------------------------------------------------------------------------
# PPO hyperparameters (shared across all experiments)
# ---------------------------------------------------------------------------

PPO_KWARGS = dict(
    policy="MlpPolicy",
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    clip_range=0.2,
    gamma=0.99,
    device="cpu",   # MlpPolicy is faster on CPU than GPU
    verbose=0,
)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(wrapper_cls=None, **wrapper_kwargs) -> gym.Env:
    """
    Create a MountainCarContinuous-v0 environment, optionally wrapped.

    Parameters
    ----------
    wrapper_cls : gym.Wrapper subclass or None
        Reward shaping wrapper to apply. None = bare sparse environment.
    **wrapper_kwargs
        Passed directly to wrapper_cls.__init__.
    """
    env = gym.make("MountainCarContinuous-v0")
    if wrapper_cls is not None:
        env = wrapper_cls(env, **wrapper_kwargs)
    env = Monitor(env)  # tracks episode rewards/lengths for SB3
    return env


# ---------------------------------------------------------------------------
# Callback: collect mean episode reward at regular intervals
# ---------------------------------------------------------------------------

class _RewardLoggerCallback(BaseCallback):
    """Records mean episode reward every `log_freq` timesteps."""

    def __init__(self, log_freq: int = 2000):
        super().__init__(verbose=0)
        self.log_freq = log_freq
        self.timesteps: list[int] = []
        self.mean_rewards: list[float] = []
        self._episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        # SB3 Monitor logs episode info in the info dict
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])

        if self.num_timesteps % self.log_freq == 0 and self._episode_rewards:
            self.timesteps.append(self.num_timesteps)
            self.mean_rewards.append(float(np.mean(self._episode_rewards[-20:])))

        return True


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    name: str,
    wrapper_cls=None,
    wrapper_kwargs: Optional[dict] = None,
    n_timesteps: int = 100_000,
    n_seeds: int = 3,
    log_freq: int = 2000,
) -> dict:
    """
    Train PPO with the given reward wrapper across multiple random seeds.

    Returns
    -------
    dict with keys:
        "name"            : experiment label
        "timesteps"       : 1-D array of evaluation timesteps
        "rewards_per_seed": list of reward arrays, one per seed
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    all_rewards: list[np.ndarray] = []
    common_timesteps = None

    for seed in range(n_seeds):
        env = make_env(wrapper_cls, **wrapper_kwargs)
        callback = _RewardLoggerCallback(log_freq=log_freq)

        model = PPO(env=env, seed=seed, **PPO_KWARGS)
        model.learn(total_timesteps=n_timesteps, callback=callback)
        env.close()

        all_rewards.append(np.array(callback.mean_rewards))
        if common_timesteps is None:
            common_timesteps = np.array(callback.timesteps)

    # Align arrays to the shortest run (in case of off-by-one across seeds)
    min_len = min(len(r) for r in all_rewards)
    all_rewards = [r[:min_len] for r in all_rewards]
    timesteps = common_timesteps[:min_len] if common_timesteps is not None else np.array([])

    return {
        "name": name,
        "timesteps": timesteps,
        "rewards_per_seed": all_rewards,
    }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_results(results: list[dict], path: str = "results.pkl") -> None:
    """Pickle a list of experiment result dicts."""
    Path(path).write_bytes(pickle.dumps(results))
    print(f"Saved {len(results)} experiment(s) → {path}")


def load_results(path: str = "results.pkl") -> list[dict]:
    """Load previously saved experiment results."""
    return pickle.loads(Path(path).read_bytes())
