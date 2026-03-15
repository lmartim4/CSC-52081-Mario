"""Multi-channel symbolic (RAM-based) grid observation for Super Mario Bros.

Builds a 13x16xC grid from NES RAM with separate channels:
  Ch 0 — Tiles:     raw metatile byte (distinguishes brick, pipe, ?, ground…)
  Ch 1 — Enemies:   enemy type + 1 (goomba, koopa, piranha… 0 = none)
  Ch 2 — Mario:     0 = absent, 1 = small, 2 = big, 3 = fire
  Ch 3 — Powerup:   0 = absent, 1 = item on screen
  Ch 4 — Fireballs: 0 = absent, 1 = fireball present

Screen positioning logic from external/yumouwei-smb/smb_utils.py (smb_grid).

References:
  - RAM map: https://datacrystal.tcrf.net/wiki/Super_Mario_Bros./RAM_map
  - Tile grid: yumouwei/super-mario-bros-reinforcement-learning
"""

import os
import sys
import gym
import numpy as np
from gym import spaces

# Make the external submodule importable
_SUBMODULE_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "external", "yumouwei-smb"
)
_SUBMODULE_DIR = os.path.abspath(_SUBMODULE_DIR)
if _SUBMODULE_DIR not in sys.path:
    sys.path.insert(0, _SUBMODULE_DIR)

from smb_utils import smb_grid  # noqa: E402

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------
VISIBLE_COLS = 16
VISIBLE_ROWS = 13

# Channel layout
N_CHANNELS = 5
CH_TILES = 0       # raw metatile byte (0 = empty, non-zero = tile type)
CH_ENEMIES = 1     # enemy type + 1 (0 = no enemy)
CH_MARIO = 2       # 0 = absent, 1 = small, 2 = big, 3 = fire
CH_POWERUP = 3     # 0 = absent, 1 = item on screen
CH_FIREBALL = 4    # 0 = absent, 1 = fireball present

# Legacy single-value constants (kept for visualization / tests)
EMPTY = 0
SOLID = 1
ENEMY = -1
MARIO = 2
POWERUP = 3
FIREBALL = 4

# ---------------------------------------------------------------------------
# NES RAM addresses
# ---------------------------------------------------------------------------
# Enemies (5 slots, offset by slot index i = 0..4)
_ENEMY_DRAWN = 0x000F
_ENEMY_TYPE = 0x0016
_ENEMY_X_PAGE = 0x006E
_ENEMY_X_SCREEN = 0x0087
_ENEMY_Y_SCREEN = 0x00CF
_MAX_ENEMIES = 5

# Mario powerup state
_MARIO_POWERUP = 0x0756   # 0 = small, 1 = big, >=2 = fire

# Powerup item on screen
_POWERUP_DRAWN = 0x0014
_POWERUP_X_SCREEN = 0x008C
_POWERUP_Y_SCREEN = 0x00D4

# Fireballs (2 slots)
# Ref: https://datacrystal.tcrf.net/wiki/Super_Mario_Bros./RAM_map
_FIREBALL_STATE = 0x0024   # 0x0024 = fireball 1, 0x0025 = fireball 2
_FIREBALL_X_SCREEN = 0x008D  # screen X position (1 byte per slot)
_FIREBALL_Y_SCREEN = 0x00D5  # screen Y position (1 byte per slot)
_MAX_FIREBALLS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tile_address(x, y):
    """Convert tile grid position (x, y) to RAM address.

    Same layout as smb_grid.tile_loc_to_ram_address.
    """
    page = x // 16
    x_loc = x % 16
    y_loc = page * 13 + y
    return 0x500 + x_loc + y_loc * 16


def grid_to_composite(grid):
    """Convert a multi-channel grid (H, W, C) to a single-channel composite.

    Useful for visualisation: maps channels back to the legacy encoding
    (EMPTY / SOLID / ENEMY / MARIO / POWERUP).
    """
    comp = np.zeros((VISIBLE_ROWS, VISIBLE_COLS), dtype=np.float32)

    # tiles: any non-zero raw value → SOLID
    comp[grid[:, :, CH_TILES] != 0] = SOLID

    # enemies (drawn over tiles)
    comp[grid[:, :, CH_ENEMIES] != 0] = ENEMY

    # powerup
    comp[grid[:, :, CH_POWERUP] != 0] = POWERUP

    # fireballs
    comp[grid[:, :, CH_FIREBALL] != 0] = FIREBALL

    # mario (always on top)
    comp[grid[:, :, CH_MARIO] != 0] = MARIO

    return comp


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------
class RAMGridObservation(gym.ObservationWrapper):
    """Multi-channel symbolic grid from NES RAM.

    Observation shape: (13, 16, 5)  float32
      Ch 0 (tiles)    : raw metatile byte (0 = empty, non-zero = tile type)
      Ch 1 (enemies)  : enemy type byte + 1 (0 = no enemy)
      Ch 2 (mario)    : 0 = absent, 1 = small, 2 = big, 3 = fire
      Ch 3 (powerup)  : 0 = absent, 1 = item visible on screen
      Ch 4 (fireball) : 0 = absent, 1 = fireball present
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(VISIBLE_ROWS, VISIBLE_COLS, N_CHANNELS),
            dtype=np.float32,
        )

    def observation(self, obs):
        ram = self.unwrapped.ram
        grid = np.zeros(
            (VISIBLE_ROWS, VISIBLE_COLS, N_CHANNELS), dtype=np.float32,
        )

        # Use smb_grid for screen-position calculations
        sg = smb_grid(self.env)
        screen_start = int(np.rint(sg.x_start / 16))

        # -- Ch 0: tiles (raw metatile value) --
        for c in range(VISIBLE_COLS):
            x_loc = (screen_start + c) % (VISIBLE_COLS * 2)
            for r in range(VISIBLE_ROWS):
                addr = _tile_address(x_loc, r)
                grid[r, c, CH_TILES] = float(ram[addr])

        # -- Ch 1: enemies (type) --
        for i in range(_MAX_ENEMIES):
            if ram[_ENEMY_DRAWN + i] != 1:
                continue
            ex = (int(ram[_ENEMY_X_PAGE + i]) * 256
                  + int(ram[_ENEMY_X_SCREEN + i])
                  - sg.x_start)
            ey = int(ram[_ENEMY_Y_SCREEN + i])
            col = int((ex + 8) // 16)
            row = int((ey + 8 - 32) // 16)
            if 0 <= row < VISIBLE_ROWS and 0 <= col < VISIBLE_COLS:
                grid[row, col, CH_ENEMIES] = float(ram[_ENEMY_TYPE + i] + 1)

        # -- Ch 2: Mario (position + powerup state) --
        mx = int((sg.mario_x + 8) // 16)
        my = int((sg.mario_y - 32) // 16)
        if 0 <= my < VISIBLE_ROWS and 0 <= mx < VISIBLE_COLS:
            p = int(ram[_MARIO_POWERUP])
            # 0→small(1), 1→big(2), >=2→fire(3)
            grid[my, mx, CH_MARIO] = float(min(p + 1, 3))

        # -- Ch 3: powerup item --
        if ram[_POWERUP_DRAWN] == 1:
            px = int(ram[_POWERUP_X_SCREEN])
            py = int(ram[_POWERUP_Y_SCREEN])
            col = int((px + 8) // 16)
            row = int((py + 8 - 32) // 16)
            if 0 <= row < VISIBLE_ROWS and 0 <= col < VISIBLE_COLS:
                grid[row, col, CH_POWERUP] = 1.0

        # -- Ch 4: fireballs (2 slots, screen-relative positions) --
        for i in range(_MAX_FIREBALLS):
            if ram[_FIREBALL_STATE + i] == 0:
                continue
            fx = int(ram[_FIREBALL_X_SCREEN + i])
            fy = int(ram[_FIREBALL_Y_SCREEN + i])
            col = int((fx + 8) // 16)
            row = int((fy + 8 - 32) // 16)
            if 0 <= row < VISIBLE_ROWS and 0 <= col < VISIBLE_COLS:
                grid[row, col, CH_FIREBALL] = 1.0

        return grid


class FlattenGrid(gym.ObservationWrapper):
    """Flatten the grid into a 1-D vector for MLP policies."""

    def __init__(self, env):
        super().__init__(env)
        flat_size = int(np.prod(self.observation_space.shape))
        self.observation_space = spaces.Box(
            low=float(env.observation_space.low.min()),
            high=float(env.observation_space.high.max()),
            shape=(flat_size,),
            dtype=np.float32,
        )

    def observation(self, obs):
        return obs.flatten()


class FrameStackGrid(gym.Wrapper):
    """Stack the last *n_stack* grids along the channel axis.

    For multi-channel input (H, W, C) → output (H, W, C * n_stack).
    For 2-D input (H, W)              → output (H, W, n_stack).
    """

    def __init__(self, env, n_stack=4, n_skip=1):
        super().__init__(env)
        self.n_stack = n_stack
        self.n_skip = n_skip
        base_shape = env.observation_space.shape

        if len(base_shape) == 3:
            H, W, C = base_shape
            stacked_shape = (H, W, C * n_stack)
        else:
            stacked_shape = (*base_shape, n_stack)

        self.observation_space = spaces.Box(
            low=float(env.observation_space.low.min()),
            high=float(env.observation_space.high.max()),
            shape=stacked_shape,
            dtype=np.float32,
        )
        self._frames = []

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._frames = [obs] * (self.n_stack * self.n_skip)
        return self._get_stacked()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        if len(self._frames) > self.n_stack * self.n_skip:
            self._frames.pop(0)
        return self._get_stacked(), reward, done, info

    def _get_stacked(self):
        indices = [
            len(self._frames) - 1 - i * self.n_skip
            for i in range(self.n_stack)
        ]
        indices = sorted(indices)
        selected = [self._frames[max(0, i)] for i in indices]

        if selected[0].ndim == 3:
            return np.concatenate(selected, axis=-1).astype(np.float32)
        return np.stack(selected, axis=-1).astype(np.float32)


class SkipFrame(gym.Wrapper):
    """Repeat the same action for *skip* frames and sum the rewards."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


# ---------------------------------------------------------------------------
# Helper: build a ready-to-use symbolic environment
# ---------------------------------------------------------------------------
def make_symbolic_env(
    env_id="SuperMarioBros-1-1-v0",
    skip=4,
    n_stack=4,
    n_skip=1,
    flatten=False,
):
    """Create a multi-channel symbolic-observation Super Mario Bros environment.

    Parameters
    ----------
    env_id : str
        gym-super-mario-bros environment id.
    skip : int
        Number of frames to repeat each action (SkipFrame wrapper).
    n_stack : int
        Number of grid frames to stack (temporal information).
        Set to 1 to disable stacking.
    n_skip : int
        Gap between stacked frames.
    flatten : bool
        If True, flatten the observation to 1-D (for MlpPolicy).

    Returns
    -------
    gym.Env
    """
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace

    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=skip)
    env = RAMGridObservation(env)

    if n_stack > 1:
        env = FrameStackGrid(env, n_stack=n_stack, n_skip=n_skip)

    if flatten:
        env = FlattenGrid(env)

    return env
