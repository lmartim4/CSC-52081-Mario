import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from typing import cast, List, Tuple, Union

import matplotlib.pyplot as plt

import seaborn as sns
from tqdm.notebook import tqdm
from IPython.display import Video, HTML
from matplotlib import animation
from ipywidgets import interact
import warnings

# warnings.filterwarnings("ignore", category=UserWarning)
# sns.set_context("talk")

# FIGS_DIR = Path("figs/") / "lab6"       # Where to save figures (.gif or .mp4 files)
# PLOTS_DIR = Path("figs/") / "lab6"      # Where to save plots (.png or .svg files)
# MODELS_DIR = Path("models/") / "lab6"   # Where to save models (.pth files)

# if not FIGS_DIR.exists():
#     FIGS_DIR.mkdir(parents=True)
# if not PLOTS_DIR.exists():
#     PLOTS_DIR.mkdir(parents=True)
# if not MODELS_DIR.exists():
#     MODELS_DIR.mkdir(parents=True)
    
def start_environment():
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")

    cartpole_observation_space = cast(gym.spaces.Box, env.observation_space)
    cartpole_action_space = cast(gym.spaces.Discrete, env.action_space)

    cartpole_observation_dim:int = cartpole_observation_space.shape[0]

    print(f"State space size is: {cartpole_observation_space}")
    print(f"Action space size is: {cartpole_action_space}")

    env.close()


def run_episode(policy_fn, max_steps=999, seed=42):
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    total_reward = 0.0
    for _ in range(max_steps):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        total_reward += float(reward)
        if terminated or truncated:
            break
    env.close()
    return frames, total_reward


def display_video(frames, fps=30):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    img = ax.imshow(frames[0])

    def update(frame):
        img.set_data(frame)
        return [img]

    anim = animation.FuncAnimation(
        fig, update, frames=frames, interval=1000 / fps, blit=True
    )
    plt.close()
    return HTML(anim.to_jshtml())