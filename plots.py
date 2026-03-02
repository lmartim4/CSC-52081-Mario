"""
Analysis and comparison plots for reward shaping experiments.

All functions accept the list of result dicts produced by experiments.run_experiment()
and return a matplotlib Figure that can be displayed inline in the notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Consistent colour palette across all plots
_PALETTE = sns.color_palette("tab10")


def _color(i: int):
    return _PALETTE[i % len(_PALETTE)]


# ---------------------------------------------------------------------------
# 1. Learning curves — convergence comparison
# ---------------------------------------------------------------------------

def plot_learning_curves(results: list[dict]) -> plt.Figure:
    """
    Mean episode reward (± 1 std across seeds) over training timesteps.
    Shows which method converges fastest and to what level.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, exp in enumerate(results):
        ts = exp["timesteps"]
        rewards = np.stack(exp["rewards_per_seed"])          # (n_seeds, T)
        mean = rewards.mean(axis=0)
        std = rewards.std(axis=0)
        color = _color(i)

        ax.plot(ts, mean, label=exp["name"], color=color, linewidth=2)
        ax.fill_between(ts, mean - std, mean + std, alpha=0.2, color=color)

    ax.axhline(90, color="gray", linestyle="--", linewidth=1, label="Success threshold (90)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Learning Curves — Reward Shaping Comparison")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax.legend(loc="lower right")
    sns.despine()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Sample efficiency — steps to reach threshold
# ---------------------------------------------------------------------------

def plot_sample_efficiency(results: list[dict], threshold: float = 90.0) -> plt.Figure:
    """
    Bar chart: median timestep at which mean reward first crosses `threshold`.
    Methods that never reach the threshold are shown as 'DNF'.
    """
    names, steps, colors = [], [], []

    for i, exp in enumerate(results):
        ts = exp["timesteps"]
        rewards = np.stack(exp["rewards_per_seed"])
        per_seed_first = []
        for r in rewards:
            idx = np.where(r >= threshold)[0]
            per_seed_first.append(ts[idx[0]] if len(idx) > 0 else np.nan)

        median_step = np.nanmedian(per_seed_first)
        names.append(exp["name"])
        steps.append(median_step)
        colors.append(_color(i))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, steps, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, steps):
        label = f"{val/1e3:.1f}k" if not np.isnan(val) else "DNF"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(s for s in steps if not np.isnan(s)) * 0.01,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel(f"Steps to reach reward ≥ {threshold:.0f}")
    ax.set_title("Sample Efficiency (lower = better)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    sns.despine()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Final performance — box plot
# ---------------------------------------------------------------------------

def plot_final_performance(results: list[dict], last_n: int = 10) -> plt.Figure:
    """
    Box plot of episode rewards in the final `last_n` logged evaluations.
    Shows the distribution of converged performance across seeds.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    data, labels, palette = [], [], []
    for i, exp in enumerate(results):
        rewards = np.stack(exp["rewards_per_seed"])        # (n_seeds, T)
        final = rewards[:, -last_n:].flatten()
        data.append(final)
        labels.append(exp["name"])
        palette.append(_color(i))

    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(90, color="gray", linestyle="--", linewidth=1, label="Success threshold (90)")
    ax.set_ylabel("Episode reward")
    ax.set_title(f"Final Performance (last {last_n} evaluations per seed)")
    ax.legend()
    sns.despine()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Training stability — reward variance over time
# ---------------------------------------------------------------------------

def plot_stability(results: list[dict]) -> plt.Figure:
    """
    Rolling std of rewards across seeds — lower means more stable training.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, exp in enumerate(results):
        ts = exp["timesteps"]
        rewards = np.stack(exp["rewards_per_seed"])        # (n_seeds, T)
        std = rewards.std(axis=0)
        ax.plot(ts, std, label=exp["name"], color=_color(i), linewidth=2)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Reward std across seeds")
    ax.set_title("Training Stability (lower = more consistent across seeds)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax.legend()
    sns.despine()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience: all-in-one summary figure
# ---------------------------------------------------------------------------

def plot_summary(results: list[dict], threshold: float = 90.0) -> plt.Figure:
    """2×2 grid with all four analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Reward Shaping Comparison — MountainCarContinuous-v0", fontsize=14)

    # Reuse individual plot logic but draw into subaxes
    _draw_learning_curves(results, axes[0, 0])
    _draw_sample_efficiency(results, axes[0, 1], threshold)
    _draw_final_performance(results, axes[1, 0])
    _draw_stability(results, axes[1, 1])

    fig.tight_layout()
    return fig


# Internal helpers for the summary plot
def _draw_learning_curves(results, ax):
    for i, exp in enumerate(results):
        ts = exp["timesteps"]
        rewards = np.stack(exp["rewards_per_seed"])
        mean, std = rewards.mean(0), rewards.std(0)
        ax.plot(ts, mean, label=exp["name"], color=_color(i), linewidth=1.8)
        ax.fill_between(ts, mean - std, mean + std, alpha=0.15, color=_color(i))
    ax.axhline(90, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Learning Curves")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Mean reward")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax.legend(fontsize=8)


def _draw_sample_efficiency(results, ax, threshold):
    names, steps, colors = [], [], []
    for i, exp in enumerate(results):
        ts = exp["timesteps"]
        rewards = np.stack(exp["rewards_per_seed"])
        per_seed = [ts[np.where(r >= threshold)[0][0]] if len(np.where(r >= threshold)[0]) > 0 else np.nan
                    for r in rewards]
        names.append(exp["name"])
        steps.append(np.nanmedian(per_seed))
        colors.append(_color(i))
    bars = ax.bar(names, steps, color=colors, edgecolor="white")
    for bar, val in zip(bars, steps):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(), f"{val/1e3:.1f}k" if not np.isnan(val) else "DNF",
                ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Steps to reward ≥ {threshold:.0f}")
    ax.set_ylabel("Timesteps")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))


def _draw_final_performance(results, ax, last_n=10):
    data = [np.stack(exp["rewards_per_seed"])[:, -last_n:].flatten() for exp in results]
    labels = [exp["name"] for exp in results]
    palette = [_color(i) for i in range(len(results))]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(90, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Final Performance")
    ax.set_ylabel("Reward")


def _draw_stability(results, ax):
    for i, exp in enumerate(results):
        ts = exp["timesteps"]
        std = np.stack(exp["rewards_per_seed"]).std(0)
        ax.plot(ts, std, label=exp["name"], color=_color(i), linewidth=1.8)
    ax.set_title("Training Stability (std)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Std across seeds")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
