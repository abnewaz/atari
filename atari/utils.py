import os
import random
import numpy as np
import torch
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_atari_env(env_name: str, frame_stack: int = 4, render_mode=None):
    """
    Create a preprocessed Atari environment:
      - NoopReset
      - Frame skipping (handled by NoFrameskip + AtariPreprocessing)
      - Grayscale + Resize to 84x84
      - Frame stacking
    """
    env = gym.make(env_name, frameskip = 1, render_mode=render_mode)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        scale_obs=True,  # scales to [0, 1]
    )
    env = FrameStackObservation(env, stack_size=frame_stack)
    return env


def discount_cumsum(rewards: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Compute discounted cumulative sums of a reward sequence.
    For Decision Transformer we typically use gamma=1.0 (undiscounted return-to-go).
    """
    disc_cumsum = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        disc_cumsum[t] = running
    return disc_cumsum


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits except the top-k."""
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_val = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)


def create_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)