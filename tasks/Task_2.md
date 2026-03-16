# Task 2 — Cache `smb_grid` Inside `RAMGridObservation`

## Problem
Every call to `RAMGridObservation.observation()` constructs a brand-new
`smb_grid` object:

```python
def observation(self, obs):
    grid = smb_grid(self.env).rendered_screen   # new object every step
```

`smb_grid.__init__` reads several RAM addresses and calls `get_rendered_screen()`
which iterates over all 16×13 = 208 tile positions plus enemy slots.
With 8 parallel environments, this happens thousands of times per second and
is one of the hottest paths in the training loop.

## What to do

Replace the one-shot construction with a **cached, reusable object** that only
re-reads the RAM on each call (which it already does internally) rather than
re-allocating Python objects.

### Option A — reuse a single instance (preferred)

Create the `smb_grid` object once in `__init__` and call a refresh method each
step instead of reconstructing:

```python
class RAMGridObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # ... observation_space setup unchanged ...
        self._grid = smb_grid(env)          # created once

    def observation(self, obs):
        self._grid.__init__(self.env)       # re-reads RAM in-place
        grid = self._grid.rendered_screen.copy()
        self._fill_powerup(grid, self.unwrapped.ram)
        return grid.astype(np.float32)
```

Check whether calling `smb_grid.__init__` again on an existing instance is safe
(it only sets instance attributes, no external side effects — it should be fine).

### Option B — inline the RAM reads

If Option A's `__init__` re-call feels fragile, copy the relevant RAM-reading
logic directly into `observation()` without the class. This is more code but
removes the object allocation entirely.

### What NOT to do
Do not cache `rendered_screen` itself across steps — it must be re-read from RAM
every step since the game state changes.

## Files to modify
- `src/wrappers/ram_wrappers.py` — `RAMGridObservation` class

## Definition of done
`RAMGridObservation.observation()` no longer creates a new `smb_grid` instance
on each call. A quick timing test (`time.perf_counter` around 1000 `env.step()`
calls) should show a measurable speedup (or at minimum no regression).
