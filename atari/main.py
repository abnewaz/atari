"""
Main entry point for training / evaluating the Decision Transformer on Breakout.

Usage:
    # Step 1: Collect offline data
    python3 main.py collect --num_episodes 1000
    
    # Step 2: Train
    python3 main.py train

    # Step 3: Evaluate
    python3 main.py eval --checkpoint checkpoints/dt_breakout_best.pt --render

    # Or do everything:
    python main.py all
"""

import argparse
import torch

from config import DTConfig
from utils import set_seed, create_dirs
from collect_data import collect_random_trajectories, save_trajectories
from train import Trainer
from evaluate import evaluate_decision_transformer
from model import DecisionTransformer


def cmd_collect(config, args):
    print("\n=== Collecting Offline Data ===\n")
    trajectories = collect_random_trajectories(config, args.num_episodes)
    save_trajectories(trajectories, config.dataset_path)


def cmd_train(config, args):
    print("\n=== Training Decision Transformer ===\n")
    trainer = Trainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


def cmd_eval(config, args):
    print("\n=== Evaluating Decision Transformer ===\n")
    device = torch.device(config.device)
    model = DecisionTransformer(config).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("WARNING: No checkpoint specified, using random weights!")

    target_return = args.target_return if args.target_return else config.target_return
    mean_ret, std_ret = evaluate_decision_transformer(
        model, config,
        target_return=target_return,
        num_episodes=args.eval_episodes,
        render=args.render,
    )
    print(f"\nEvaluation Result: {mean_ret:.1f} ± {std_ret:.1f}")


def cmd_all(config, args):
    """Collect data, train, and evaluate."""
    cmd_collect(config, args)
    cmd_train(config, args)
    # Evaluate best checkpoint
    args.checkpoint = "checkpoints/dt_breakout_best.pt"
    args.render = False
    cmd_eval(config, args)


def main():
    parser = argparse.ArgumentParser(description="Decision Transformer for Breakout")
    subparsers = parser.add_subparsers(dest="command")

    # ── collect ──
    p_collect = subparsers.add_parser("collect", help="Collect offline data")
    p_collect.add_argument("--num_episodes", type=int, default=1000)

    # ── train ──
    p_train = subparsers.add_parser("train", help="Train the model")
    p_train.add_argument("--resume", type=str, default=None,
                         help="Path to checkpoint to resume from")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--batch_size", type=int, default=None)
    p_train.add_argument("--lr", type=float, default=None)

    # ── eval ──
    p_eval = subparsers.add_parser("eval", help="Evaluate the model")
    p_eval.add_argument("--checkpoint", type=str, default=None)
    p_eval.add_argument("--target_return", type=float, default=None)
    p_eval.add_argument("--eval_episodes", type=int, default=10)
    p_eval.add_argument("--render", action="store_true")

    # ── all ──
    p_all = subparsers.add_parser("all", help="Collect, train, and evaluate")
    p_all.add_argument("--num_episodes", type=int, default=1000)

    args = parser.parse_args()

    # Build config (override with CLI args if provided)
    config = DTConfig()
    if hasattr(args, "epochs") and args.epochs:
        config.epochs = args.epochs
    if hasattr(args, "batch_size") and args.batch_size:
        config.batch_size = args.batch_size
    if hasattr(args, "lr") and args.lr:
        config.learning_rate = args.lr
    if hasattr(args, "eval_episodes") and hasattr(args, "eval_episodes"):
        pass  # handled in cmd_eval

    set_seed(config.seed)

    if args.command == "collect":
        cmd_collect(config, args)
    elif args.command == "train":
        cmd_train(config, args)
    elif args.command == "eval":
        cmd_eval(config, args)
    elif args.command == "all":
        cmd_all(config, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()