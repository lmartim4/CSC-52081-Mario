"""
Test: Record side-by-side video of base env (pixels) vs RAM wrapper (multi-channel grid).

Runs 5 episodes and saves an MP4 with:
  - Left:   pixel render from the NES emulator
  - Right:  composite grid (tiles + enemies + Mario + powerups)

Run from the project root:
    conda run -n mario python tests/test_video_comparison.py
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from src.wrappers.ram_wrappers import (
    RAMGridObservation, SkipFrame, grid_to_composite,
    EMPTY, SOLID, ENEMY, MARIO, POWERUP,
    VISIBLE_ROWS, VISIBLE_COLS,
    CH_TILES, CH_ENEMIES, CH_MARIO, CH_POWERUP,
)

# Composite colormap: enemy=red, empty=skyblue, solid=brown, mario=lime, powerup=gold
COMP_CMAP = mcolors.ListedColormap(["red", "skyblue", "saddlebrown", "lime", "gold"])
COMP_BOUNDS = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
COMP_NORM = mcolors.BoundaryNorm(COMP_BOUNDS, COMP_CMAP.N)

SYMBOLS = {EMPTY: "", SOLID: "#", ENEMY: "E", MARIO: "M", POWERUP: "?"}


def render_frame(pixel_obs, grid_obs, info, episode, step):
    """Render one comparison frame as an RGB numpy array."""
    composite = grid_to_composite(grid_obs)

    fig = Figure(figsize=(12, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # -- Left: pixel render --
    ax_px = fig.add_subplot(1, 3, 1)
    ax_px.imshow(pixel_obs)
    ax_px.set_title(
        f"Pixels | ep={episode} step={step}\nx={info.get('x_pos', '?')}",
        fontsize=8,
    )
    ax_px.axis("off")

    # -- Middle: composite grid (like old single-channel view) --
    ax_comp = fig.add_subplot(1, 3, 2)
    ax_comp.imshow(composite, cmap=COMP_CMAP, norm=COMP_NORM, interpolation="nearest")
    ax_comp.set_title("Composite Grid", fontsize=8)
    ax_comp.set_xticks(range(VISIBLE_COLS))
    ax_comp.set_yticks(range(VISIBLE_ROWS))
    ax_comp.tick_params(labelsize=4)
    ax_comp.grid(True, color="gray", linewidth=0.3, alpha=0.3)
    for r in range(VISIBLE_ROWS):
        for c in range(VISIBLE_COLS):
            val = int(composite[r, c])
            sym = SYMBOLS.get(val, "")
            if sym:
                color = "white" if val in (SOLID, ENEMY) else "black"
                ax_comp.text(
                    c, r, sym, ha="center", va="center",
                    fontsize=4, fontweight="bold", color=color,
                )

    # -- Right: per-channel detail --
    ch_names = ["Tiles (raw)", "Enemies", "Mario", "Powerup"]
    ch_cmaps = ["YlOrBr", "Reds", "Greens", "cool"]
    ax_ch = fig.add_subplot(1, 3, 3)
    ax_ch.axis("off")

    # Create 2x2 sub-grid inside the right panel
    from mpl_toolkits.axes_grid1 import ImageGrid
    sub_fig = fig.add_gridspec(1, 3)[2].subgridspec(2, 2, wspace=0.3, hspace=0.4)

    for idx in range(4):
        ax = fig.add_subplot(sub_fig[idx // 2, idx % 2])
        ch_data = grid_obs[:, :, idx]
        ax.imshow(ch_data, cmap=ch_cmaps[idx], interpolation="nearest")
        ax.set_title(ch_names[idx], fontsize=5)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    canvas.draw()
    buf = canvas.buffer_rgba()
    frame = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return frame


def main():
    output_path = os.path.join(PROJECT_ROOT, "results", "comparison_video.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_episodes = 5
    skip = 4
    actions_sequence = [1, 1, 2, 1, 1, 1, 2, 1]

    # Base env (pixels)
    env_raw = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env_raw = JoypadSpace(env_raw, SIMPLE_MOVEMENT)
    env_raw = SkipFrame(env_raw, skip=skip)

    # RAM wrapper env (multi-channel grid, no frame stack)
    env_sym = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env_sym = JoypadSpace(env_sym, SIMPLE_MOVEMENT)
    env_sym = SkipFrame(env_sym, skip=skip)
    env_sym = RAMGridObservation(env_sym)

    frames = []
    print(f"Recording {n_episodes} episodes...")

    for episode in range(1, n_episodes + 1):
        env_raw.reset()
        env_sym.reset()
        done = False
        step = 0

        while not done:
            action = actions_sequence[step % len(actions_sequence)]
            pixel_obs, _, d1, info = env_raw.step(action)
            grid_obs, _, d2, _ = env_sym.step(action)
            step += 1
            done = d1 or d2

            frame = render_frame(pixel_obs, grid_obs, info, episode, step)
            frames.append(frame)

        print(f"  Episode {episode}: {step} steps, x_pos={info.get('x_pos', '?')}")

    env_raw.close()
    env_sym.close()

    print(f"\nTotal frames: {len(frames)}")
    print(f"Saving video to {output_path} ...")
    imageio.mimsave(output_path, frames, fps=15, codec="libx264")
    print(f"[OK] Video saved: {output_path}")


if __name__ == "__main__":
    main()
