"""
PyTorch Dataset that samples sub-sequences of length K from offline trajectories.
Each sample returns:
    - states:        (K, C, H, W) image observations
    - actions:       (K,)          discrete actions
    - returns_to_go: (K,)          undiscounted return-to-go at each step
    - timesteps:     (K,)          absolute timestep within the episode
    - attention_mask:(K,)          1 for real tokens, 0 for padding
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class BreakoutTrajectoryDataset(Dataset):
    def __init__(self, dataset_path: str, context_length: int, max_ep_len: int):
        super().__init__()
        self.context_length = context_length
        self.max_ep_len = max_ep_len

        with open(dataset_path, "rb") as f:
            self.trajectories = pickle.load(f)

        # Compute sampling weights proportional to trajectory length
        # so longer trajectories are sampled more often
        self.lengths = np.array([t["length"] for t in self.trajectories])
        self.total_transitions = int(self.lengths.sum())
        self.p_sample = self.lengths / self.lengths.sum()

        # Pre-sort trajectories by return (useful for logging)
        returns = [t["total_return"] for t in self.trajectories]
        sorted_inds = np.argsort(returns)
        self.sorted_trajectories = [self.trajectories[i] for i in sorted_inds]

        print(f"Loaded {len(self.trajectories)} trajectories, "
              f"{self.total_transitions} total transitions.")
        print(f"  Returns — min: {min(returns):.1f}, max: {max(returns):.1f}, "
              f"mean: {np.mean(returns):.1f}")

    def __len__(self):
        # We define "epoch" as sampling total_transitions sequences
        return self.total_transitions

    def __getitem__(self, idx):
        K = self.context_length

        # Sample a trajectory weighted by length
        traj_idx = np.random.choice(len(self.trajectories), p=self.p_sample)
        traj = self.trajectories[traj_idx]
        traj_len = traj["length"]

        # Sample a random starting index within the trajectory
        if traj_len <= K:
            start = 0
        else:
            start = np.random.randint(0, traj_len - K + 1)

        end = min(start + K, traj_len)
        actual_len = end - start

        # Extract sub-sequence
        states = traj["observations"][start:end]       # (actual_len, C, H, W)
        actions = traj["actions"][start:end]            # (actual_len,)
        returns_to_go = traj["returns_to_go"][start:end]  # (actual_len,)
        timesteps = np.arange(start, end)               # (actual_len,)

        # Pad to context_length K if necessary
        pad_len = K - actual_len

        if pad_len > 0:
            state_shape = states.shape[1:]  # (C, H, W)
            states = np.concatenate([
                np.zeros((pad_len, *state_shape), dtype=np.float32),
                states
            ], axis=0)
            actions = np.concatenate([
                np.zeros(pad_len, dtype=np.int64),
                actions
            ])
            returns_to_go = np.concatenate([
                np.zeros(pad_len, dtype=np.float32),
                returns_to_go
            ])
            timesteps = np.concatenate([
                np.zeros(pad_len, dtype=np.int64),
                timesteps
            ])
            attention_mask = np.concatenate([
                np.zeros(pad_len, dtype=np.float32),
                np.ones(actual_len, dtype=np.float32)
            ])
        else:
            attention_mask = np.ones(K, dtype=np.float32)

        # Normalize returns-to-go (scale by some factor for stability)
        # Typical max score in Breakout is ~400+; we'll normalize by 100
        returns_to_go = returns_to_go / 100.0

        # Clip timesteps to max_ep_len
        timesteps = np.clip(timesteps, 0, self.max_ep_len - 1)

        return {
            "states": torch.tensor(states, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.long),
            "returns_to_go": torch.tensor(returns_to_go, dtype=torch.float32),
            "timesteps": torch.tensor(timesteps, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
        }