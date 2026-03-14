"""Wrapper for symbolic (RAM-based) grid observation.

Builds a compact 13x16 grid directly from the NES RAM, encoding tiles,
enemies, and Mario's position — no pixel processing needed.

References:
  - RAM map: https://datacrystal.tcrf.net/wiki/Super_Mario_Bros./RAM_map
  - Tile grid: yumouwei/super-mario-bros-reinforcement-learning (smb_utils.py)
  - Enemy logic: Chrispresso/SuperMarioBros-AI (utils.py)
"""

import gym
import numpy as np
from gym import spaces

# ---------------------------------------------------------------------------
# NES RAM addresses for Super Mario Bros.
# Reference: https://datacrystal.tcrf.net/wiki/Super_Mario_Bros./RAM_map
# ---------------------------------------------------------------------------

# --- Tiles ---
# The level tiles are stored as two 16×13 pages (= 32×13 total) at 0x0500.
# Each byte != 0 is a "solid" tile (block, pipe, ground, brick, etc.).
TILE_RAM_START = 0x0500
TILE_RAM_END = 0x069F  # inclusive → 416 bytes = 32 cols × 13 rows
TILE_COLS_TOTAL = 32  # two 16-col pages side by side
TILE_ROWS = 13

# --- Screen scroll ---
# Which 256-px "page" of the level the screen is currently showing.
CURRENT_SCREEN_PAGE = 0x071A
# How many pixels into that page the screen has scrolled (0–255).
SCREEN_X_SCROLL = 0x071C

# --- Player ---
PLAYER_X_POS_SCREEN = 0x0086  # x position on screen (pixels)
PLAYER_Y_POS_SCREEN = 0x00CE  # y position on screen (pixels)
PLAYER_STATE = 0x000E  # 0x08 = normal, 0x06/0x0B = dying, etc.
PLAYER_POWERUP = 0x0756  # 0=small, 1=big, >=2=fire
PLAYER_FLOAT = 0x001D  # 0=ground, 1=jumping, 2=falling

# --- Enemies (5 slots: addr + 0..4) ---
ENEMY_DRAWN = 0x000F  # 5 bytes: 1 = active on screen
ENEMY_TYPE = 0x0016  # 5 bytes: enemy type id
ENEMY_X_POS_SCREEN = 0x0087  # 5 bytes: x on screen (pixels)
ENEMY_Y_POS_SCREEN = 0x00CF  # 5 bytes: y on screen (pixels)
ENEMY_X_PAGE = 0x006E  # 5 bytes: horizontal position in level

MAX_ENEMIES = 5

# --- Powerup ---
POWERUP_DRAWN = 0x0014  # 1 = powerup on screen
POWERUP_X_SCREEN = 0x008C
POWERUP_Y_SCREEN = 0x00D4

# Grid encoding values
EMPTY = 0
SOLID = 1
ENEMY = -1
MARIO = 2
POWERUP = 3

# Visible grid dimensions (what the screen shows)
VISIBLE_COLS = 16
VISIBLE_ROWS = 13


class RAMGridObservation(gym.ObservationWrapper):
    """Convert the pixel observation into a 13×16 symbolic grid read from RAM.

    The grid encodes:
       0 = empty / sky
       1 = solid tile (ground, brick, pipe, block, etc.)
      -1 = enemy
       2 = Mario
       3 = powerup / item

    The observation is a float32 array of shape (13, 16).
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-1, high=3,
            shape=(VISIBLE_ROWS, VISIBLE_COLS),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observation(self, obs):
        """Ignore the pixel obs; build grid from RAM instead."""
        ram = self.unwrapped.ram
        return self._build_grid(ram).astype(np.float32)

    # ------------------------------------------------------------------
    # Internal: grid construction
    # ------------------------------------------------------------------

    def _build_grid(self, ram):
        """Build a 13×16 symbolic grid from the NES RAM."""
        grid = np.zeros((VISIBLE_ROWS, VISIBLE_COLS), dtype=np.int32)

        # 1) Read tile data ------------------------------------------------
        self._fill_tiles(grid, ram)

        # 2) Place enemies --------------------------------------------------
        self._fill_enemies(grid, ram)

        # 3) Place powerups -------------------------------------------------
        self._fill_powerup(grid, ram)

        # 4) Place Mario ----------------------------------------------------
        self._fill_mario(grid, ram)

        return grid

    # --- Tiles -----------------------------------------------------------

    def _fill_tiles(self, grid, ram):
        """Read the 32×13 tile buffer and extract the 16-col visible window.

        The NES stores tiles in two 16×13 pages at 0x0500.  The screen
        scrolls across these pages; we compute which 16 columns are
        currently visible and copy them into the grid.
        """
        # Full 32×13 tile buffer (column-major in RAM: each column is 13 bytes)
        full_tiles = np.zeros((TILE_ROWS, TILE_COLS_TOTAL), dtype=np.int32)
        for col in range(TILE_COLS_TOTAL):
            for row in range(TILE_ROWS):
                addr = TILE_RAM_START + col * TILE_ROWS + row
                full_tiles[row, col] = ram[addr]

        # Which column of the 32-col buffer corresponds to the left edge
        # of the screen?  The screen page (0x071A) tells us which 256-px
        # page we are on; each tile is 16 px wide, so one page = 16 tile cols.
        # The scroll register (0x071C) gives the pixel offset within that page.
        page = int(ram[CURRENT_SCREEN_PAGE])
        scroll_px = int(ram[SCREEN_X_SCROLL])
        start_col = (page % 2) * 16 + scroll_px // 16

        for vc in range(VISIBLE_COLS):
            src_col = (start_col + vc) % TILE_COLS_TOTAL
            for row in range(TILE_ROWS):
                if full_tiles[row, src_col] != 0:
                    grid[row, vc] = SOLID

    # --- Enemies ----------------------------------------------------------

    @staticmethod
    def _fill_enemies(grid, ram):
        """Place active enemies onto the grid."""
        for i in range(MAX_ENEMIES):
            if ram[ENEMY_DRAWN + i] == 0:
                continue

            ex = int(ram[ENEMY_X_POS_SCREEN + i])
            ey = int(ram[ENEMY_Y_POS_SCREEN + i])

            col = ex // 16
            row = (ey - 32) // 16  # subtract 32 px for the status bar

            if 0 <= row < VISIBLE_ROWS and 0 <= col < VISIBLE_COLS:
                grid[row, col] = ENEMY

    # --- Powerup ----------------------------------------------------------

    @staticmethod
    def _fill_powerup(grid, ram):
        """Place a powerup (mushroom / flower / star) on the grid."""
        if ram[POWERUP_DRAWN] == 0:
            return

        px = int(ram[POWERUP_X_SCREEN])
        py = int(ram[POWERUP_Y_SCREEN])

        col = px // 16
        row = (py - 32) // 16

        if 0 <= row < VISIBLE_ROWS and 0 <= col < VISIBLE_COLS:
            grid[row, col] = POWERUP

    # --- Mario ------------------------------------------------------------

    @staticmethod
    def _fill_mario(grid, ram):
        """Place Mario on the grid (always last so he is visible)."""
        mx = int(ram[PLAYER_X_POS_SCREEN])
        my = int(ram[PLAYER_Y_POS_SCREEN])

        col = mx // 16
        row = (my - 32) // 16

        if 0 <= row < VISIBLE_ROWS and 0 <= col < VISIBLE_COLS:
            grid[row, col] = MARIO


class FlattenGrid(gym.ObservationWrapper):
    """Flatten the 2-D grid into a 1-D vector for MLP policies."""

    def __init__(self, env):
        super().__init__(env)
        flat_size = int(np.prod(self.observation_space.shape))
        self.observation_space = spaces.Box(
            low=-1, high=3,
            shape=(flat_size,),
            dtype=np.float32,
        )

    def observation(self, obs):
        return obs.flatten()


class FrameStackGrid(gym.Wrapper):
    """Stack the last *n_stack* grid frames along a new last axis.

    Resulting shape: (13, 16, n_stack).
    This gives the agent temporal information (velocity, direction).
    """

    def __init__(self, env, n_stack=4, n_skip=1):
        super().__init__(env)
        self.n_stack = n_stack
        self.n_skip = n_skip
        base_shape = env.observation_space.shape  # (13, 16)
        self.observation_space = spaces.Box(
            low=-1, high=3,
            shape=(*base_shape, n_stack),
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
        # Pick every n_skip-th frame from the buffer
        indices = [
            len(self._frames) - 1 - i * self.n_skip
            for i in range(self.n_stack)
        ]
        indices = sorted(indices)
        selected = [self._frames[max(0, i)] for i in indices]
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
    """Create a symbolic-observation Super Mario Bros environment.

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
        If False, return shape (13, 16, n_stack).

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
