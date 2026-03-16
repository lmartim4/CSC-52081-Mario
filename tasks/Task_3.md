# Task 3 — Curriculum Callback for `random_start_steps`

## Problem
`RandomStartWrapper` uses a **fixed** `max_start_steps=100` for the entire
training run. Early in training the agent has no skills — being dropped at
step 80 with enemies nearby produces pure noise gradients. The agent should
earn harder starting positions as it improves.

## What to do

### 1. Add `set_max_start_steps` to `RandomStartWrapper`

In `src/wrappers/ram_wrappers.py`, add a setter method so the callback can
update the value at runtime:

```python
def set_max_start_steps(self, value: int):
    self.max_start_steps = int(value)
```

SB3's `SubprocVecEnv` exposes `env_method(method_name, *args)` which forwards
a method call to every sub-environment across the subprocess boundary.

### 2. Create `CurriculumCallback` in `src/utils/callbacks.py`

```python
class CurriculumCallback(BaseCallback):
    """Linearly increase random_start_steps as training progresses."""

    def __init__(self, start_steps=0, end_steps=100, total_timesteps=4_000_000):
        super().__init__()
        self.start_steps = start_steps
        self.end_steps = end_steps
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        fraction = min(self.n_calls / self.total_timesteps, 1.0)
        current = int(self.start_steps + fraction * (self.end_steps - self.start_steps))
        self.training_env.env_method('set_max_start_steps', current)
        return True
```

Call `env_method` every step is fine — it's a trivial integer assignment in
each subprocess. If performance is a concern, only call it every N steps
(e.g. `if self.n_calls % 1000 == 0`).

### 3. Export from `src/utils/__init__.py`

Add `CurriculumCallback` to the imports in `src/utils/__init__.py` (if that
file re-exports callbacks).

### 4. Update notebook 11

In `notebooks/11_world1_random_start_train.ipynb`, import and use it:

```python
from src.utils.callbacks import CheckpointAndLogCallback, CurriculumCallback
from stable_baselines3.common.callbacks import CallbackList

checkpoint_cb = CheckpointAndLogCallback(
    save_path='models/symbolic_ppo_world1_random',
    save_freq=100_000,
)
curriculum_cb = CurriculumCallback(
    start_steps=0,
    end_steps=100,
    total_timesteps=TOTAL_STEPS,
)

model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=CallbackList([checkpoint_cb, curriculum_cb]),
    log_interval=1,
)
```

Also set `random_start_steps=0` in `make_symbolic_multitask_vec_env` (the
curriculum starts at 0 and handles ramping) — or keep `random_start_steps=100`
as a hard cap and let the curriculum ramp from 0 to 100.

## Files to modify
- `src/wrappers/ram_wrappers.py` — add `set_max_start_steps` to `RandomStartWrapper`
- `src/utils/callbacks.py` — add `CurriculumCallback`
- `notebooks/11_world1_random_start_train.ipynb` — wire it in

## Definition of done
During a short training run, printing `max_start_steps` from inside the env
confirms it increases from 0 toward 100 over the course of training.
