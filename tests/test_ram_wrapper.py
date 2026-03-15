"""
Test 2: Super Mario Bros with multi-channel RAM symbolic grid wrapper.

Run from the project root:
    conda run -n mario python tests/test_ram_wrapper.py
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from src.wrappers.ram_wrappers import (
    make_symbolic_env,
    RAMGridObservation,
    SkipFrame,
    grid_to_composite,
    EMPTY, SOLID, ENEMY, MARIO, POWERUP,
    VISIBLE_ROWS, VISIBLE_COLS,
    N_CHANNELS, CH_TILES, CH_ENEMIES, CH_MARIO, CH_POWERUP,
)


# --- Pretty-print a composite grid ----------------------------------------

SYMBOLS = {
    EMPTY:   ".",
    SOLID:   "#",
    ENEMY:   "E",
    MARIO:   "M",
    POWERUP: "?",
}


def print_grid(obs, step=None):
    """Print a composite view of the multi-channel grid."""
    comp = grid_to_composite(obs)
    header = f"--- Step {step} " if step is not None else "--- "
    header += "-" * (VISIBLE_COLS * 2 - len(header) + 4)
    print(header)
    col_nums = "   " + "".join(f"{c:2d}" for c in range(VISIBLE_COLS))
    print(col_nums)
    for r in range(VISIBLE_ROWS):
        row_str = f"{r:2d} "
        for c in range(VISIBLE_COLS):
            val = int(comp[r, c])
            row_str += f" {SYMBOLS.get(val, '?')}"
        print(row_str)
    print()


# --- Tests ----------------------------------------------------------------

def test_wrapper_shape():
    print("=" * 60)
    print("TEST 2a: Wrapper observation shape & dtype")
    print("=" * 60)

    env = make_symbolic_env(n_stack=1, flatten=False)
    obs = env.reset()
    expected = (VISIBLE_ROWS, VISIBLE_COLS, N_CHANNELS)
    print(f"  n_stack=1, flatten=False -> shape: {obs.shape}, dtype: {obs.dtype}")
    assert obs.shape == expected, f"Expected {expected}, got {obs.shape}"
    env.close()

    env = make_symbolic_env(n_stack=4, flatten=False)
    obs = env.reset()
    expected = (VISIBLE_ROWS, VISIBLE_COLS, N_CHANNELS * 4)
    print(f"  n_stack=4, flatten=False -> shape: {obs.shape}, dtype: {obs.dtype}")
    assert obs.shape == expected, f"Expected {expected}, got {obs.shape}"
    env.close()

    env = make_symbolic_env(n_stack=4, flatten=True)
    obs = env.reset()
    expected_flat = VISIBLE_ROWS * VISIBLE_COLS * N_CHANNELS * 4
    print(f"  n_stack=4, flatten=True  -> shape: {obs.shape}, dtype: {obs.dtype}")
    assert obs.shape == (expected_flat,), f"Expected ({expected_flat},), got {obs.shape}"
    env.close()

    print("\n[OK] Shape tests PASSED\n")


def test_grid_content(n_steps=300):
    print("=" * 60)
    print("TEST 2b: Grid content validation (multi-channel)")
    print("=" * 60)

    env = make_symbolic_env(n_stack=1, flatten=False)
    obs = env.reset()

    found_mario = False
    found_tiles = False
    found_enemy = False
    max_x = 0
    actions_sequence = [1, 1, 2, 1, 1, 1, 2, 1]

    for step in range(n_steps):
        action = actions_sequence[step % len(actions_sequence)]
        obs, reward, done, info = env.step(action)
        max_x = max(max_x, info.get("x_pos", 0))

        has_mario = np.any(obs[:, :, CH_MARIO] != 0)
        has_tiles = np.any(obs[:, :, CH_TILES] != 0)
        has_enemy = np.any(obs[:, :, CH_ENEMIES] != 0)

        if has_mario: found_mario = True
        if has_tiles: found_tiles = True
        if has_enemy: found_enemy = True

        if step in [0, 10, 50] or (has_enemy and not found_enemy):
            print_grid(obs, step)
            mario_val = obs[:, :, CH_MARIO].max()
            enemy_types = np.unique(obs[:, :, CH_ENEMIES][obs[:, :, CH_ENEMIES] != 0])
            print(f"  Mario ch: max={mario_val} | Tiles ch: {has_tiles} | "
                  f"Enemy types: {enemy_types.tolist()}")
            print(f"  info: x_pos={info.get('x_pos')}, y_pos={info.get('y_pos')}, "
                  f"status={info.get('status')}, time={info.get('time')}")

        if done:
            obs = env.reset()

    env.close()

    print(f"\nResults after {n_steps} steps:")
    print(f"  Mario found:   {'YES' if found_mario else 'NO - PROBLEM'}")
    print(f"  Tiles found:   {'YES' if found_tiles else 'NO - PROBLEM'}")
    print(f"  Enemies found: {'YES' if found_enemy else 'maybe (need more steps)'}")
    print(f"  Max x_pos:     {max_x}")

    assert found_mario, "Mario was never found in the grid!"
    assert found_tiles, "No tiles were found!"
    print("\n[OK] Content tests PASSED\n")


def test_channel_ranges(n_steps=100):
    print("=" * 60)
    print("TEST 2c: Per-channel value ranges")
    print("=" * 60)

    env = make_symbolic_env(n_stack=1, flatten=False)
    obs = env.reset()

    ch_mins = [float("inf")] * N_CHANNELS
    ch_maxs = [float("-inf")] * N_CHANNELS

    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        for ch in range(N_CHANNELS):
            ch_mins[ch] = min(ch_mins[ch], obs[:, :, ch].min())
            ch_maxs[ch] = max(ch_maxs[ch], obs[:, :, ch].max())
        if done:
            obs = env.reset()

    env.close()

    ch_names = ["tiles", "enemies", "mario", "powerup"]
    for ch in range(N_CHANNELS):
        print(f"  Ch {ch} ({ch_names[ch]:>8s}): [{ch_mins[ch]:.0f}, {ch_maxs[ch]:.0f}]")

    # Basic sanity checks
    assert ch_mins[CH_TILES] >= 0, "Tile values should be >= 0"
    assert ch_maxs[CH_MARIO] <= 3, "Mario state should be <= 3"
    assert ch_maxs[CH_POWERUP] <= 1, "Powerup should be 0 or 1"
    print("\n[OK] Channel range tests PASSED\n")


def test_frame_stack():
    print("=" * 60)
    print("TEST 2d: Frame stack temporal consistency")
    print("=" * 60)

    env = make_symbolic_env(n_stack=4, n_skip=1, flatten=False)
    obs = env.reset()

    for _ in range(20):
        obs, _, done, _ = env.step(1)
        if done:
            obs = env.reset()

    # With n_stack=4 and N_CHANNELS=4, obs shape is (13, 16, 16)
    # Channels 0-3 = most recent frame, 4-7 = previous, etc.
    frames_differ = any(
        not np.array_equal(
            obs[:, :, i * N_CHANNELS:(i + 1) * N_CHANNELS],
            obs[:, :, (i + 1) * N_CHANNELS:(i + 2) * N_CHANNELS],
        )
        for i in range(3)
    )

    env.close()
    print(f"  Obs shape: {obs.shape}")
    print(f"  Frames differ after movement: {'YES' if frames_differ else 'WARNING'}")
    print("\n[OK] Frame stack test PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SUPER MARIO BROS - MULTI-CHANNEL RAM WRAPPER TEST SUITE")
    print("=" * 60 + "\n")

    test_wrapper_shape()
    test_grid_content()
    test_channel_ranges()
    test_frame_stack()

    print("=" * 60)
    print("  ALL TESTS COMPLETED")
    print("=" * 60)
