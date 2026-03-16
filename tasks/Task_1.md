# Task 1 — Add Mario's Power State to the Observation

## Problem
`RAMGridObservation` builds a 13×16 symbolic grid of tiles, enemies, and Mario's
position, but it never encodes **whether Mario is small, super, or fire**.

This means the agent cannot distinguish situations where touching an enemy
is lethal (small Mario) from situations where it is safe to stomp (super Mario).
The optimal policy is fundamentally different in each state, but the network
sees identical observations.

## What to do

### 1. Read the power state from RAM

RAM address `0x0756` holds Mario's power state:
- `0` = small
- `1` = super (big)
- `2` = fire

It is accessible via `self.unwrapped.ram[0x0756]` anywhere in the wrapper stack.

### 2. Append it to the flattened observation

The cleanest place is **`FlattenGrid`** in
[src/wrappers/ram_wrappers.py](../src/wrappers/ram_wrappers.py).

After flattening the 13×16 grid (832 values), append the power state as a
single normalised float (divide by 2 so it sits in [0, 1], matching the rest).

The observation space must grow from `(832,)` to `(833,)`.

Update `FlattenGrid.__init__` to set `flat_size = np.prod(...) + 1` and update
the `observation_space` bounds accordingly.

Update `FlattenGrid.observation` to read the RAM value and concatenate.

`FlattenGrid` currently has no direct reference to the unwrapped env.
Access it via `self.unwrapped` (gym's wrapper chain supports this).

### 3. Update observation space bounds

The new element is in `[0, 1]` (normalised). The existing grid values are in
`[-1, 3]`. You can either:
- Keep a single `Box(low=-1, high=3, shape=(833,))` — slightly imprecise but harmless.
- Use per-element bounds with `np.concatenate` on `low` and `high` arrays — more correct.

### 4. Verify nothing else breaks

- `make_symbolic_env` with `flatten=True` produces shape `(833,)` ✓
- `make_symbolic_vec_env` and `make_symbolic_multitask_vec_env` inherit this automatically ✓
- Notebook 11 uses `flatten=True`, so observation space becomes `(833,)`. The PPO
  model definition in that notebook will automatically use the new size since it
  reads from `env.observation_space.shape`. ✓
- **Pre-trained models** (`models/symbolic_ppo/final_model`) were trained on `(832,)`
  and **cannot** be loaded into an `(833,)` env without retraining. Note this in a
  comment in the notebook.

## Files to modify
- `src/wrappers/ram_wrappers.py` — `FlattenGrid` class

## Definition of done
`make_symbolic_env(flatten=True).observation_space.shape == (833,)` and the
power state value changes visibly when Mario picks up a mushroom during a test run.
