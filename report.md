# Training Diagnostics Report

## Critical Issues

- **Replay sampling can return missing experiences** – `random.sample(range(1, replay_memory_size + 1), ...)` includes the value `replay_memory_size`, but your Redis keys go from 0 to `replay_memory_size - 1`. When that top value is sampled, `r.get(key)` returns `None` and `pickle.loads` raises a `TypeError`, which stops training. This also happens whenever the buffer holds fewer than `BATCH_SIZE` items because `random.sample` rejects a request larger than the population (raising `ValueError`). See `train.py:132-141`.
- **Device hard-coded to "mps" during batching** – The training loop creates tensors with `device="mps"`, ignoring the computed `device` variable. On CPUs (or CUDA GPUs) that line throws `RuntimeError: "mps" not available`, so the learner never runs. See `train.py:143-147`.

## High Impact Logic Gaps

- **Distance reward ignores prior distance and step penalty** – The reward shaping in `Snake.move_with_action` computes `self.distance_reward = 1/(self.new_distance + 1)` without subtracting the previous distance (the code that tracks `self.current_distance` is commented out). Because the agent gets a positive reward no matter what, moving away from food still gives a (smaller) positive return, and the `step_penalty` is never applied because every call uses the default `step_count=0`. This makes the learning signal very weak and biases Q-values toward simply staying alive. See `game.py:269-309`.

## Additional Observations

- **Target network never frozen** – Assigning `target_nn.requires_grad_ = False` overwrites the method instead of disabling gradients. The target network still tracks gradients, which is unnecessary work and breaks future calls to `requires_grad_`. Use `target_nn.requires_grad_(False)` instead. See `train.py:169-170`.
- **Replay key 0 is never sampled** – Keys start at 0 (`EPISODE_ID = 0` in `data.py:65-83`), but the sampling range starts at 1. This leaves the very first experience unused. Not critical, but easy to fix by sampling `range(replay_memory_size)` and converting to strings.

## Suggested Next Steps

1. Fix the replay sampler to only request existing keys and guard against buffers smaller than `BATCH_SIZE`.
2. Switch tensor device placement to the computed `device` variable (and broadcast tensors via `.to(device)`).
3. Reinstate the distance-delta reward (or another shaping term) and feed a non-zero `step_count` so step penalties matter.
4. Freeze the target network properly with `target_nn.requires_grad_(False)`.
