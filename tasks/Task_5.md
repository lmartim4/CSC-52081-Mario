# Task 5 — Load Pre-Trained 1-1 Model in Notebook 11

## Problem
Notebook 11 (`notebooks/11_world1_random_start_train.ipynb`) trains a PPO agent
from scratch on all four World 1 levels simultaneously. This forces the agent
to rediscover basic locomotion (running, jumping) at the same time as handling
level variety and random starting positions — a much harder learning problem.

A pre-trained 1-1 model already exists at `models/symbolic_ppo/final_model`
(produced by notebook 09). It already knows how to run and jump. Fine-tuning
from this base is faster and produces a stronger agent.

## Important caveat — observation space must match

**Check Task 1 first.** If Task 1 has been completed, the observation space
is now `(833,)` (power state appended) but `models/symbolic_ppo/final_model`
was trained on `(832,)`. These are **incompatible** — `PPO.load` will raise a
shape mismatch error.

- **If Task 1 is NOT done**: observation is `(832,)` on both sides — load directly.
- **If Task 1 IS done**: the pre-trained model cannot be loaded as-is. Either:
  - Retrain notebook 09 to produce a `(833,)`-compatible base model, OR
  - Train from scratch in notebook 11 (this task becomes a no-op for now).

## What to do (when observation spaces match)

### Replace cell 4 in notebook 11

Current cell 4 creates a model from scratch with `PPO(...)`.

Replace it with:

```python
TOTAL_STEPS = 4_000_000

model = PPO.load(
    'models/symbolic_ppo/final_model',
    env=env,
    device='cpu',
    # Override hyperparams for multi-task fine-tuning
    learning_rate=1e-4,          # lower than initial 2.5e-4 — we're fine-tuning
    ent_coef=0.02,               # higher entropy for exploration on new levels
)

print(f'Loaded pre-trained 1-1 model')
print(f'Fine-tuning: {TOTAL_STEPS:,} steps across all World 1 levels')
print(f'Device: {model.device}')
```

### Rationale for hyperparameter changes
- **`learning_rate=1e-4`** (vs original `2.5e-4`): lower rate for fine-tuning
  avoids catastrophic forgetting of 1-1 skills while adapting to new levels.
- **`ent_coef=0.02`** (vs original `0.01`): higher entropy encourages exploration
  from unfamiliar random-start positions in levels 1-2, 1-3, 1-4.

### Update the save path and log dir

The fine-tuned model should save separately from the base:
```python
tensorboard_log='logs/symbolic_ppo_world1_random'     # already correct in nb11
save_path='models/symbolic_ppo_world1_random'          # already correct in nb11
```

### Update the markdown cell (cell 0)

Add a note that this notebook fine-tunes from `models/symbolic_ppo/final_model`
rather than training from scratch.

## Files to modify
- `notebooks/11_world1_random_start_train.ipynb` — cell 4 (model creation)
  and cell 0 (markdown description)

## Definition of done
The notebook loads cleanly without shape errors, TensorBoard shows reward
starting from a non-zero baseline (reflecting pre-trained 1-1 knowledge)
rather than from the random-policy floor.
