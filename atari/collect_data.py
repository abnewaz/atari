# """
# Collect offline trajectories for Breakout.

# Strategy:
#   - Use a mix of random and partially-trained agents.
#   - Each trajectory stores: observations, actions, rewards, dones, returns-to-go.
#   - Save as a pickle file.

# You can also substitute this with the DQN Replay Dataset from
# https://research.google/tools/datasets/dqn-replay/ or d4rl-atari.
# """

# import os
# import pickle
# import argparse
# import numpy as np
# from tqdm import tqdm

# from config import DTConfig
# from utils import make_atari_env, discount_cumsum, set_seed, create_dirs


# def collect_random_trajectories(config: DTConfig, num_episodes: int):
#     """Collect trajectories using a random policy."""
#     env = make_atari_env(config.env_name, config.frame_stack)
#     trajectories = []

#     for ep in tqdm(range(num_episodes), desc="Collecting random trajectories"):
#         obs_list, act_list, rew_list, done_list = [], [], [], []

#         obs, _ = env.reset(seed=config.seed + ep)
#         done = False
#         truncated = False

#         while not (done or truncated):
#             action = env.action_space.sample()
#             next_obs, reward, done, truncated, info = env.step(action)

#             # obs is a LazyFrames object; convert to numpy
#             obs_list.append(np.array(obs, dtype=np.float32))
#             act_list.append(action)
#             rew_list.append(reward)
#             done_list.append(done or truncated)

#             obs = next_obs

#         observations = np.array(obs_list, dtype=np.float32)   # (T, 4, 84, 84)
#         actions = np.array(act_list, dtype=np.int64)           # (T,)
#         rewards = np.array(rew_list, dtype=np.float32)         # (T,)
#         dones = np.array(done_list, dtype=bool)                # (T,)
#         returns_to_go = discount_cumsum(rewards, gamma=1.0)    # (T,)

#         trajectory = {
#             "observations": observations,
#             "actions": actions,
#             "rewards": rewards,
#             "dones": dones,
#             "returns_to_go": returns_to_go,
#             "total_return": float(rewards.sum()),
#             "length": len(rewards),
#         }
#         trajectories.append(trajectory)

#     env.close()
#     return trajectories


# def save_trajectories(trajectories, path):
#     create_dirs(os.path.dirname(path))
#     with open(path, "wb") as f:
#         pickle.dump(trajectories, f)
#     print(f"Saved {len(trajectories)} trajectories to {path}")
#     returns = [t["total_return"] for t in trajectories]
#     print(f"  Return stats — mean: {np.mean(returns):.1f}, "
#           f"max: {np.max(returns):.1f}, min: {np.min(returns):.1f}")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--num_episodes", type=int, default=1000)
#     parser.add_argument("--output", type=str, default=None)
#     args = parser.parse_args()

#     config = DTConfig()
#     set_seed(config.seed)

#     output_path = args.output or config.dataset_path
#     trajectories = collect_random_trajectories(config, args.num_episodes)
#     save_trajectories(trajectories, output_path)


# if __name__ == "__main__":
#     main()

"""
Collect offline trajectories for Breakout.

Strategy:
  - Use a mix of random and partially-trained agents.
  - Each trajectory stores: observations, actions, rewards, dones, returns-to-go.
  - Save as a pickle file.

You can also substitute this with the DQN Replay Dataset from
https://research.google/tools/datasets/dqn-replay/ or d4rl-atari.
"""

import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from config import DTConfig
from utils import make_atari_env, discount_cumsum, set_seed, create_dirs


def collect_single_episode(ep_id, config):
    """Collect a single random episode (for parallel execution)."""
    env = make_atari_env(config.env_name, config.frame_stack)
    obs_list, act_list, rew_list, done_list = [], [], [], []

    obs, _ = env.reset(seed=config.seed + ep_id)
    done = False
    truncated = False

    while not (done or truncated):
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)

        obs_list.append(np.array(obs, dtype=np.float32))
        act_list.append(action)
        rew_list.append(reward)
        done_list.append(done or truncated)

        obs = next_obs

    env.close()

    observations = np.array(obs_list, dtype=np.float32)
    actions = np.array(act_list, dtype=np.int64)
    rewards = np.array(rew_list, dtype=np.float32)
    dones = np.array(done_list, dtype=bool)
    returns_to_go = discount_cumsum(rewards, gamma=1.0)

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "returns_to_go": returns_to_go,
        "total_return": float(rewards.sum()),
        "length": len(rewards),
    }


def collect_random_trajectories(config: DTConfig, num_episodes: int, num_workers: int = 8):
    """Collect trajectories using a random policy (parallel version)."""
    print(f"Collecting {num_episodes} episodes using {num_workers} workers...")
    
    with mp.Pool(num_workers) as pool:
        collect_fn = partial(collect_single_episode, config=config)
        trajectories = list(tqdm(
            pool.imap(collect_fn, range(num_episodes)),
            total=num_episodes,
            desc="Collecting random trajectories"
        ))
    
    return trajectories


def collect_in_batches(config: DTConfig, total_episodes: int, batch_size: int = 2000, num_workers: int = 8):
    """
    Collect episodes in batches and save incrementally to avoid memory/time issues.
    """
    num_batches = (total_episodes + batch_size - 1) // batch_size
    all_trajectories = []
    
    for batch_idx in range(num_batches):
        start_ep = batch_idx * batch_size
        end_ep = min(start_ep + batch_size, total_episodes)
        current_batch_size = end_ep - start_ep
        
        print(f"\n=== Batch {batch_idx + 1}/{num_batches} ===")
        print(f"Collecting episodes {start_ep} to {end_ep - 1} ({current_batch_size} episodes)")
        
        # Collect batch
        batch_trajectories = collect_random_trajectories(
            config, 
            num_episodes=current_batch_size,
            num_workers=num_workers
        )
        
        # Save batch immediately (checkpoint)
        batch_path = f"data/breakout_batch_{batch_idx}.pkl"
        save_trajectories(batch_trajectories, batch_path)
        
        all_trajectories.extend(batch_trajectories)
        
        print(f"Progress: {len(all_trajectories)}/{total_episodes} episodes collected")
        
    return all_trajectories


def save_trajectories(trajectories, path):
    create_dirs(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Saved {len(trajectories)} trajectories to {path}")
    returns = [t["total_return"] for t in trajectories]
    print(f"  Return stats — mean: {np.mean(returns):.1f}, "
          f"max: {np.max(returns):.1f}, min: {np.min(returns):.1f}")


def combine_batches(output_path: str = "data/breakout_trajectories.pkl"):
    """Combine all batch files into one dataset."""
    import glob
    
    batch_files = sorted(glob.glob("data/breakout_batch_*.pkl"))
    if not batch_files:
        print("No batch files found!")
        return
    
    print(f"\nCombining {len(batch_files)} batch files...")
    all_trajectories = []
    
    for batch_file in batch_files:
        with open(batch_file, "rb") as f:
            batch = pickle.load(f)
            all_trajectories.extend(batch)
            print(f"  Loaded {len(batch)} trajectories from {batch_file}")
    
    save_trajectories(all_trajectories, output_path)
    print(f"\nCombined dataset saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2000, help="Save checkpoint every N episodes")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--combine", action="store_true", help="Combine existing batch files")
    args = parser.parse_args()

    config = DTConfig()
    set_seed(config.seed)

    if args.combine:
        combine_batches(args.output or config.dataset_path)
        return

    output_path = args.output or config.dataset_path
    
    # Collect in batches with checkpointing
    trajectories = collect_in_batches(
        config,
        total_episodes=args.num_episodes,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Save final combined dataset
    save_trajectories(trajectories, output_path)


if __name__ == "__main__":
    main()