# Task 4 — Per-Level Evaluation During Training

## Problem
`CheckpointAndLogCallback` aggregates episode rewards from all 8 environments
into a single metric. With 4 levels mixed, a rising mean reward could mean the
agent is improving on 1-1 while stagnating on 1-4 — or vice versa. There is
no way to tell.

## What to do

### 1. Add a `PerLevelEvalCallback` to `src/utils/callbacks.py`

This callback runs deterministic evaluation episodes on each level separately
at a fixed interval and logs the results to TensorBoard.

```python
class PerLevelEvalCallback(BaseCallback):
    """Periodically evaluate the model on each level and log separately."""

    def __init__(self, levels, eval_freq=100_000, n_eval_episodes=5,
                 skip=4, n_stack=4):
        super().__init__()
        self.levels = levels          # list of env_id strings
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.skip = skip
        self.n_stack = n_stack

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        from src.wrappers import make_symbolic_env
        import numpy as np

        for level in self.levels:
            env = make_symbolic_env(
                env_id=level, skip=self.skip, n_stack=self.n_stack,
                flatten=True, random_start_steps=0,
            )
            rewards, flags, x_positions = [], [], []

            for _ in range(self.n_eval_episodes):
                result = env.reset()
                obs = result[0] if isinstance(result, tuple) else result
                done, total_reward, flag, max_x = False, 0.0, False, 0

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    result = env.step(int(action))
                    if len(result) == 5:
                        obs, reward, terminated, truncated, info = result
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = result
                    total_reward += float(reward)
                    max_x = max(max_x, info.get('x_pos', 0))
                    if info.get('flag_get', False):
                        flag = True

                rewards.append(total_reward)
                flags.append(flag)
                x_positions.append(max_x)

            env.close()

            # Use a short tag, e.g. "1-1", "1-2" from the env_id
            tag = level.split('-')[1] + '-' + level.split('-')[2]
            self.logger.record(f'eval/{tag}_mean_reward', np.mean(rewards))
            self.logger.record(f'eval/{tag}_flag_rate',   np.mean(flags))
            self.logger.record(f'eval/{tag}_mean_x_pos',  np.mean(x_positions))

        self.logger.dump(self.num_timesteps)
        return True
```

### 2. Wire it into notebook 11

```python
from src.utils.callbacks import (
    CheckpointAndLogCallback,
    PerLevelEvalCallback,
)
from stable_baselines3.common.callbacks import CallbackList

checkpoint_cb = CheckpointAndLogCallback(
    save_path='models/symbolic_ppo_world1_random',
    save_freq=100_000,
)
eval_cb = PerLevelEvalCallback(
    levels=LEVELS,
    eval_freq=200_000,
    n_eval_episodes=5,
)

model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=CallbackList([checkpoint_cb, eval_cb]),
    log_interval=1,
)
```

### Notes
- Eval runs **without** random starts (`random_start_steps=0`) so results are
  comparable across checkpoints.
- `n_eval_episodes=5` is a lightweight choice — increase for more stable metrics.
- The eval loop runs in the **main process** (not a subprocess), so it adds
  wall-clock time proportional to `n_eval_episodes × level_count` at each
  `eval_freq` checkpoint. Keep `eval_freq` large enough to not dominate runtime.

## Files to modify
- `src/utils/callbacks.py` — add `PerLevelEvalCallback`
- `notebooks/11_world1_random_start_train.ipynb` — wire it in

## Definition of done
TensorBoard shows separate `eval/1-1_*`, `eval/1-2_*`, `eval/1-3_*`,
`eval/1-4_*` curves updating every `eval_freq` steps.
