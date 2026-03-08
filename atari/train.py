"""
Training script for the Decision Transformer on Breakout.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from config import DTConfig
from model import DecisionTransformer
from dataset import BreakoutTrajectoryDataset
from evaluate import evaluate_decision_transformer
from utils import set_seed, create_dirs


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


import math


class Trainer:
    def __init__(self, config: DTConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Model
        self.model = DecisionTransformer(config).to(self.device)

        # Dataset & DataLoader
        self.dataset = BreakoutTrajectoryDataset(
            config.dataset_path, config.context_length, config.max_ep_len
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # LR Scheduler
        total_steps = config.epochs * len(self.dataloader)
        self.scheduler = get_lr_scheduler(
            self.optimizer, config.warmup_steps, total_steps
        )

        # Loss
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

        # Logging
        create_dirs(config.log_dir, config.save_dir)
        self.global_step = 0
        self.best_eval_return = -float("inf")

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0

        for batch in self.dataloader:
            states = batch["states"].to(self.device)                # (B, K, C, H, W)
            actions = batch["actions"].to(self.device)              # (B, K)
            returns_to_go = batch["returns_to_go"].to(self.device)  # (B, K)
            timesteps = batch["timesteps"].to(self.device)          # (B, K)
            attention_mask = batch["attention_mask"].to(self.device) # (B, K)

            # Forward
            action_logits = self.model(
                states, actions, returns_to_go, timesteps, attention_mask
            )  # (B, K, n_actions)

            # Compute loss only on valid (non-padded) positions
            B, K, A = action_logits.shape
            logits_flat = action_logits.reshape(-1, A)     # (B*K, A)
            targets_flat = actions.reshape(-1)              # (B*K,)
            mask_flat = attention_mask.reshape(-1)          # (B*K,)

            loss_per_token = self.loss_fn(logits_flat, targets_flat)  # (B*K,)
            loss = (loss_per_token * mask_flat).sum() / mask_flat.sum()

            # Accuracy
            preds = logits_flat.argmax(dim=-1)
            correct = ((preds == targets_flat).float() * mask_flat).sum()
            total_correct += correct.item()
            total_tokens += mask_flat.sum().item()

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = total_correct / max(total_tokens, 1)
        lr = self.scheduler.get_last_lr()[0]

        print(f"  [Train] Epoch {epoch} | Loss: {avg_loss:.4f} | "
              f"Acc: {accuracy:.4f} | LR: {lr:.6f} | Steps: {self.global_step}")

        return avg_loss

    def save_checkpoint(self, epoch: int, tag: str = "latest"):
        path = os.path.join(self.config.save_dir, f"dt_breakout_{tag}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }, path)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["global_step"]
        print(f"  Loaded checkpoint from epoch {ckpt['epoch']}: {path}")
        return ckpt["epoch"]

    def train(self):
        print("=" * 60)
        print("Decision Transformer — Training on Breakout")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Context length K: {self.config.context_length}")
        print(f"Embed dim: {self.config.embed_dim}")
        print(f"Layers: {self.config.n_layers}, Heads: {self.config.n_heads}")
        print("=" * 60)

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            elapsed = time.time() - t0
            print(f"  Epoch {epoch} completed in {elapsed:.1f}s")

            # Evaluate periodically
            if epoch % self.config.eval_every == 0:
                mean_return, std_return = evaluate_decision_transformer(
                    self.model, self.config
                )
                print(f"  [Eval] Mean Return: {mean_return:.1f} ± {std_return:.1f}")

                if mean_return > self.best_eval_return:
                    self.best_eval_return = mean_return
                    self.save_checkpoint(epoch, tag="best")

            # Save latest checkpoint every epoch
            self.save_checkpoint(epoch, tag="latest")

        print("Training complete!")