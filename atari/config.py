import dataclasses
from typing import Optional


@dataclasses.dataclass
class DTConfig:
    """Configuration for the Decision Transformer on Atari Breakout."""

    # ---------- Environment ----------
    env_name: str = "ALE/Breakout-v5"
    frame_stack: int = 4
    image_size: int = 84

    # ---------- Dataset / Offline Data ----------
    dataset_path: str = "data/breakout_trajectories.pkl"
    num_trajectories: int = 1000          # number of trajectories to collect
    data_collection_episodes: int = 50    # episodes per random/pretrained agent run

    # ---------- Sequence / Context ----------
    context_length: int = 30   # K in the paper (number of timesteps in context)
    max_ep_len: int = 10_000   # maximum episode length (for timestep embedding)

    # ---------- Model ----------
    n_heads: int = 8
    n_layers: int = 6
    embed_dim: int = 128
    dropout: float = 0.1
    activation: str = "gelu"

    # ---------- Image Encoder (CNN) ----------
    cnn_channels: tuple = (32, 64, 64)
    cnn_kernels: tuple = (8, 4, 3)
    cnn_strides: tuple = (4, 2, 1)
    cnn_output_dim: int = 128  # linear projection after CNN

    # ---------- Training ----------
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 6e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 10_000
    grad_clip: float = 1.0
    num_workers: int = 4

    # ---------- Evaluation ----------
    eval_episodes: int = 10
    target_return: float = 90.0   # conditioning target return-to-go
    eval_every: int = 5           # evaluate every N epochs
    device: str = "cuda"           # "cpu", "cuda", "mps"

    # ---------- Logging ----------
    log_dir: str = "logs"
    save_dir: str = "checkpoints"
    seed: int = 42