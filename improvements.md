## Overview

This document captures concrete improvements to make the DQN agent learn Snake reliably. Each section explains:

- Why the issue matters
- How it manifests in the current codebase
- What to change (step-by-step)
- Ready-to-paste code snippets


## Experience storage and sampling (Redis)

- **Why it matters**: DQN relies on a diverse, uniformly sampled replay buffer. If experiences overwrite each other or sampling is biased, training becomes unstable or stalls.
- **Current behavior**: In `data.py`, transitions are stored with a pickled initial state as the Redis key:
  - `r.set(pickle.dumps(initial_state), pickle.dumps(experience))`
  - States can repeat, overwriting prior entries. `redis.randomkey()` in `train.py` can return non-experience keys and does not yield uniform sampling over experiences.
- **Fix**:
  - Use a monotonically increasing integer ID to define unique keys like `exp:{id}`.
  - Maintain a counter (e.g., `exp:id`) via `INCR`.
  - Sample by ID range, not by `RANDOMKEY`.

### Writer (data collection) change

```python
# At top-level (data.py)
EXP_ID_KEY = "exp:id"

# When writing an experience
exp_id = r.incr(EXP_ID_KEY)
key = f"exp:{exp_id}"
# Store full tuple: (state, action, reward, done, next_state)
experience = (initial_state, action, reward, done, final_state)
r.set(key, pickle.dumps(experience))
```

### Reader (training) change

```python
# In train.py, replace RANDOMKEY sampling with ID sampling
import random

def sample_batch_from_redis(r, batch_size):
    # Total experiences written so far
    max_id = int(r.get("exp:id") or 0)
    if max_id == 0:
        return []
    # Uniformly sample IDs; retry if some keys missing
    candidate_ids = random.sample(range(1, max_id + 1), k=min(batch_size, max_id))
    keys = [f"exp:{i}" for i in candidate_ids]
    raw = r.mget(keys)
    batch = [pickle.loads(x) for x in raw if x]
    return batch
```


## Reward sparsity and data collection policy

- **Why it matters**: Purely random play in Snake rarely eats food. With `EPSILON = 1` and no decay during collection, the dataset is dominated by 0/−1 rewards. The agent sees too few positive returns, preventing value propagation.
- **Fix**:
  - Decay epsilon during collection to mix random exploration with a gradually improving policy.
  - Interleave brief training phases during collection (collect N steps, train M steps), so the policy improves and collects better data.
  - Incorporate shaped rewards (once the bug below is fixed) to provide dense learning signals.

### Practical epsilon schedule (collection)

```python
eps_start = 1.0
eps_end = 0.05
eps_decay_steps = 200_000  # tune

def epsilon_by_step(global_step):
    frac = min(1.0, global_step / eps_decay_steps)
    return eps_start + frac * (eps_end - eps_start)
```


## Mini-batch training (not single samples)

- **Why it matters**: Training on a single random transition per step yields extremely noisy gradients and poor convergence. DQN expects mini-batches with averaged loss.
- **Fix**:
  - Sample batches of size 32–256 (64 is a good start) from Redis.
  - Vectorize the loss computation using `gather` to select Q(s, a).

### Batched DQN loss (standard)

```python
def compute_dqn_loss(active_nn, target_nn, batch, gamma, device):
    # batch: list of (state, action, reward, done, next_state)
    states, actions, rewards, dones, next_states = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)

    q_values = active_nn(states)                    # [B, A]
    q_selected = q_values.gather(1, actions)        # [B, 1]

    with torch.no_grad():
        next_q_values = target_nn(next_states)      # [B, A]
        next_q_max = next_q_values.max(dim=1, keepdim=True).values  # [B, 1]
        targets = rewards + (1.0 - dones) * gamma * next_q_max

    loss = torch.mean((targets - q_selected) ** 2)
    return loss
```


## State preprocessing (normalization and encoding)

- **Why it matters**: Mixed-scale inputs (grid coordinates 0–40/30, direction as an index, and binary features) impede learning. Normalization and one-hot encodings help.
- **Fix**:
  - Normalize positions by grid dimensions.
  - Replace `dir_idx` scalar with one-hot of length 4.
  - Keep dangers as 0/1. New input size becomes `2 (head) + 2 (food) + 4 (dir) + 3 (dangers) = 11`.

### Preprocess helper

```python
def preprocess_state(raw_state, grid_w=40, grid_h=30):
    head_x, head_y, dir_idx, food_x, food_y, danger_st, danger_r, danger_l = raw_state

    head_x = head_x / grid_w
    head_y = head_y / grid_h
    food_x = food_x / grid_w
    food_y = food_y / grid_h

    dir_one_hot = [0.0, 0.0, 0.0, 0.0]
    dir_one_hot[int(dir_idx)] = 1.0

    return [head_x, head_y, food_x, food_y] + dir_one_hot + [float(danger_st), float(danger_r), float(danger_l)]
```

Use during data collection and training for both `state` and `next_state`.


## Target computation with no_grad

- **Why it matters**: Targets must be treated as constants to avoid backpropagating through the target network and to reduce memory usage.
- **Fix**: Wrap next-state evaluation in `with torch.no_grad():` and build targets outside the autograd graph.

### Safer loss body (single step)

```python
active_prediction = active_nn(state_tensor)[0]  # [A]

with torch.no_grad():
    if done:
        q_target = torch.tensor([reward], device=device)
    else:
        next_q = target_nn(next_state_tensor)[0]  # [A]
        q_target = torch.tensor([reward], device=device) + gamma * next_q.max().unsqueeze(0)

loss = (q_target - active_prediction[action].unsqueeze(0)) ** 2
```


## Environment reward shaping bug (distance reward)

- **Why it matters**: The current code computes `self.new_distance` using the head position from before the move, so the distance delta is incorrect, typically penalizing steps even when moving closer to food.
- **Fix**: Recompute head position after `self.move()` and then calculate the distance difference.

### Corrected snippet in `Snake.move_with_action`

```python
# Before move
food_x, food_y = self.get_food_position()
head_x, head_y = self.get_head_position()
self.current_distance = abs(food_x - head_x) + abs(food_y - head_y)

# Execute move
reward = self.move()

# After move: recompute head position and distance
head_x, head_y = self.get_head_position()
self.new_distance = abs(food_x - head_x) + abs(food_y - head_y)
self.distance_reward = (self.current_distance - self.new_distance) / (GRID_WIDTH + GRID_HEIGHT)
```

Notes:
- Flip sign to be positive when getting closer: `(current - new)`.
- Combine with a small step penalty if desired: `self.distance_reward -= step_penalty * step_count`.
- If you choose to include this shaped reward in training, store it as the `reward` of the transition (or add to immediate reward) during collection.


## Network architecture simplification

- **Why it matters**: `DQN_scaled_ff` stacks many Linear layers without activations. This adds depth without nonlinearity and hinders optimization.
- **Fix**: Start with a 2–3 layer MLP with ReLU. Example:

```python
class DQNSmall(nn.Module):
    def __init__(self, input_size, output_size, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, output_size),
        )
    def forward(self, x):
        return self.net(x)
```

You can later add LayerNorm/Dropout if needed, but begin simple.


## Replay sampling via RANDOMKEY

- **Why it matters**: `RANDOMKEY` can return non-experience keys and does not ensure uniform sampling over the dataset. It also depends on internal Redis key distribution.
- **Fix**: Replace `redis.randomkey()` with ID-based sampling (see above). Namespace all experience keys with a prefix `exp:` to avoid collisions.


## Additional recommended training hygiene

- **Target network update frequency**: Update every 1–5k gradient steps rather than 20 epochs, especially when switching to mini-batch updates.
- **Replay buffer size**: 50k–1M transitions typically suffice. `15,500,000` is excessive and slow to manage. Start at `100_000`.
- **Optimizer and LR**: `Adam/AdamW` with `1e-3` is fine; consider `Huber loss` for robustness:

```python
huber = nn.SmoothL1Loss()
loss = huber(q_selected, targets)
```

- **Gradient clipping**: Helps stability:

```python
torch.nn.utils.clip_grad_norm_(active_nn.parameters(), max_norm=10.0)
```

- **Reward clipping**: Optional but common in DQN: clip rewards to `[-1, 1]`.
- **Double DQN**: Reduces overestimation bias (see below).
- **Training loop budget**: Replace `EPOCHS = 1_000_000` with a step/episode-driven loop, e.g., train for N environment steps (e.g., 5–20M), logging every K steps.


## Double DQN target (optional but helpful)

- **Why it matters**: Standard DQN overestimates Q-values by using the same network to select and evaluate actions.
- **Fix**: Use active network to select argmax action on next state, but evaluate its value using the target network.

```python
with torch.no_grad():
    next_q_online = active_nn(next_states)               # [B, A]
    next_actions = next_q_online.argmax(dim=1, keepdim=True)  # [B, 1]
    next_q_target = target_nn(next_states)               # [B, A]
    next_q_selected = next_q_target.gather(1, next_actions)   # [B, 1]
    targets = rewards + (1.0 - dones) * gamma * next_q_selected
```


## Logging and diagnostics (W&B)

- **Track exploration**: Log epsilon during collection and training.
- **Dataset sanity checks**:
  - Histogram of rewards in replay (expect some positives after decay starts).
  - Fraction of terminal transitions in batch.
- **Training metrics**:
  - Average Q-values and TD-error (loss) per step.
  - Evaluation score/steps at fixed intervals with `epsilon=0`.

Example W&B logs during training:

```python
wandb.log({
    "global_step": global_step,
    "epsilon": epsilon,
    "loss": float(loss.item()),
    "avg_q": float(q_values.mean().item()),
})
```


## Action space consistency

- **Observation**: `train.py` uses `ACTION_SPACE = [0,1,2,3,4]` while `agent.py` shows `[0,1,2,3]`. Ensure the environment and policy agree on the mapping:
  - `0: straight`, `1: up`, `2: down`, `3: left`, `4: right` (per `game.py`).
- **Fix**: Remove unused action spaces and keep a single source of truth imported by both data collection and training.


## Tensor shapes and device handling

- **Batch dimension**: When moving to batches, ensure inputs are `[batch, features]`. For single-state evaluation, `state_tensor.unsqueeze(0)` ensures 2D shape.
- **Device**: Construct tensors directly on the target device to avoid repeated transfers; keep dtype consistent (`float32` for states, `long` for actions).

```python
states = torch.as_tensor(states, dtype=torch.float32, device=device)
actions = torch.as_tensor(actions, dtype=torch.long, device=device)
```


## Putting it together: minimal stable training loop sketch

```python
batch_size = 64
target_update_every = 1000
global_step = 0

while global_step < max_env_steps:
    # Collect one step (with epsilon policy) and write to Redis
    epsilon = epsilon_by_step(global_step)
    # ... step env, store (state, action, reward, done, next_state)

    # Train every K steps once buffer is warm
    if r.get("exp:id") and int(r.get("exp:id")) > 10_000:
        batch = sample_batch_from_redis(r, batch_size)
        if not batch:
            continue
        loss = compute_dqn_loss(active_nn, target_nn, batch, GAMMA, device)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(active_nn.parameters(), 10.0)
        optimizer.step()

        if global_step % target_update_every == 0:
            target_nn.load_state_dict(active_nn.state_dict())

    global_step += 1
```


## Checklist

- [ ] Switch Redis storage to `exp:{id}` keys with `INCR` counter
- [ ] Replace `RANDOMKEY` with uniform ID sampling and `MGET`
- [ ] Implement mini-batch training (64) with vectorized loss
- [ ] Normalize state and one-hot encode direction (update `INPUT_SIZE`)
- [ ] Wrap target computations with `no_grad`
- [ ] Fix distance reward bug and optionally use shaped reward
- [ ] Simplify network to 2–3 layers with ReLU
- [ ] Tune target update frequency, replay size, and add gradient clipping
- [ ] Consider Double DQN targets
- [ ] Log epsilon, reward distribution, avg Q, evaluation score
- [ ] Ensure action space is consistent across files


