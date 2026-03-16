# CSC-52081 — Super Mario Bros RL

Comparing **PPO with pixel observations (CnnPolicy)** vs **PPO with symbolic/RAM observations (MlpPolicy)** on Super Mario Bros, including transfer learning between levels and analysis of catastrophic forgetting.

## Project Overview

### Motivation

Traditional approaches to Super Mario Bros use **Dueling DQN** with Double DQN and Prioritized Experience Replay. We trained a vanilla DQN as a baseline but it failed to converge in 200k steps — confirming that DQN requires significant extensions for this domain.

We chose **PPO** (Proximal Policy Optimization) as an alternative: it is simpler, on-policy, avoids Q-value overestimation, and is more stable to train. Our main research question: **can PPO with a compact symbolic representation (RAM) compete with pixel-based approaches?**

### Approach

We compare two observation representations under the same PPO algorithm and reward shaping:

| | Pixel (CnnPolicy) | Symbolic/RAM (MlpPolicy) |
|---|---|---|
| **Observation** | Grayscale 84x84, 4 frame stack | 13x16 grid from NES RAM, 4 frame stack, flattened (832-dim) |
| **Network** | Nature CNN (Mnih et al. 2015) | MLP [512, 512] |
| **Speed** | ~78 fps (Colab T4 GPU) | ~500 fps (CPU, 8 cores) |
| **Preprocessing** | Max-pooling (anti-flickering), grayscale, resize | RAM grid extraction via `smb_grid` |

**Reward shaping** (identical for both, from vietnh1009):
- Score bonus: `+score_delta / 40`
- Flag reached: `+50`
- Death / timeout: `-50`
- All rewards normalized by `/10`

**Hyperparameters**: gamma=0.9, lr=2.5e-4 (Phase 1) → 1e-5 (Phase 2), 8 parallel envs (SubprocVecEnv), batch_size=256, n_steps=512, n_epochs=4.

### Key Results

- **Symbolic PPO 1-1**: 100% flag rate, reward ~315, converged in ~500k steps (~17 min)
- **Pixel PPO 1-1**: Trained on Colab T4 GPU (significantly slower)
- **Transfer learning 1-1 → 1-2**: Successfully learned 1-2 (100% flag rate), but suffered **catastrophic forgetting** (0% on 1-1)
- **Multi-task training** (1-1 + 1-2 simultaneously): The agent converged only on 1-1 due to reward imbalance — the easier level dominates the gradient
- **Symbolic is much more sample-efficient** than pixel — faster training, faster convergence, comparable final performance

### Experiments

1. **DQN baseline** — vanilla DQN fails on Mario (motivates PPO)
2. **Pixel PPO** — CnnPolicy with vietnh1009's optimized wrappers
3. **Symbolic PPO** — MlpPolicy with RAM-based grid observation (our main contribution)
4. **Transfer learning** — fine-tune 1-1 model on 1-2
5. **Catastrophic forgetting analysis** — 1-2 model fails on 1-1
6. **Multi-task attempt** — training on both levels simultaneously (negative result, discussed)

## Setup

```bash
conda activate mario_lucas  # or your environment
pip install -r requirements.txt
```

### Apply compatibility patches

The dependencies `gym`, `nes-py`, and `gym-super-mario-bros` require patching for NumPy 2.0:

```bash
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
patch -p0 $SITE/nes_py/_rom.py                   < patches/nes_py_numpy2.patch
patch -p0 $SITE/gym/utils/passive_env_checker.py < patches/gym_bool8_numpy2.patch
patch -p0 $SITE/gym_super_mario_bros/smb_env.py  < patches/smb_env_numpy2.patch
patch -p1 $SITE/gym/wrappers/time_limit.py       < patches/gym_time_limit_compat.patch
```

```fish
set SITE (python -c "import site; print(site.getsitepackages()[0])")
patch -p0 $SITE/nes_py/_rom.py                   < patches/nes_py_numpy2.patch
patch -p0 $SITE/gym/utils/passive_env_checker.py < patches/gym_bool8_numpy2.patch
patch -p0 $SITE/gym_super_mario_bros/smb_env.py  < patches/smb_env_numpy2.patch
patch -p1 $SITE/gym/wrappers/time_limit.py       < patches/gym_time_limit_compat.patch
```


## Usage

### Training notebooks

| Notebook | Description |
|---|---|
| `notebooks/07_custom_ddqn_train.ipynb` | PPO with pixel observations (Colab GPU) |
| `notebooks/08_symbolic_dqn_train.ipynb` | PPO with symbolic (RAM) observations (local CPU) |
| `notebooks/09_evaluate_checkpoints.ipynb` | Checkpoint evaluation & learning curves |
| `notebooks/10_multitask_train.ipynb` | Multi-task training on 1-1 + 1-2 |

### Watch a trained agent play

```bash
# Symbolic agent on World 1-1
python watch_agent.py --model ../models/symbolic_ppo/final_model --env SuperMarioBros-1-1-v3

# Symbolic agent on World 1-2 (transfer learned)
python watch_agent.py --model ../models/symbolic_ppo_1_2/final_model --env SuperMarioBros-1-2-v3

# Pixel agent on World 1-1
python watch_agent.py --model ../models/pixel_ppo_v2/final_model --env SuperMarioBros-1-1-v3 --pixel
```

### TensorBoard

```bash
tensorboard --logdir ../logs/symbolic_ppo   # symbolic training
tensorboard --logdir ../logs/pixel_ppo      # pixel training
tensorboard --logdir ../logs               # all runs
```

## Project Structure

```
src/
├── wrappers/
│   ├── pixel_wrappers.py    # CustomReward, CustomSkipFrame, make_pixel_vec_env
│   └── ram_wrappers.py      # CustomRewardRAM, RAMGridObservation, make_symbolic_vec_env
├── utils/
│   ├── callbacks.py         # CheckpointAndLogCallback (tracks episodes without Monitor)
│   ├── evaluation.py        # evaluate_agent, run_episode
│   └── smb_utils.py         # RAM grid extraction (from yumouwei)
├── agents/                  # Agent implementations
└── config.py                # PPOConfig, ENV_ID, etc.

external/
└── yumouwei-smb/            # RAM-based grid submodule

notebooks/                   # Training & evaluation notebooks
models/                      # Saved checkpoints & final models
logs/                        # TensorBoard logs
patches/                     # NumPy 2.0 compatibility patches
watch_agent.py               # Visual evaluation with pyglet
```

## References

- Schulman et al. (2017) — Proximal Policy Optimization Algorithms
- Mnih et al. (2015) — Human-level control through deep reinforcement learning (Nature CNN)
- Wang et al. (2016) — Dueling Network Architectures for Deep Reinforcement Learning
- vietnh1009 — Super-mario-bros-PPO-pytorch (reward shaping & pixel wrappers)
- yumouwei — super-mario-bros-reinforcement-learning (RAM grid extraction)
- Raffin et al. — Stable-Baselines3
