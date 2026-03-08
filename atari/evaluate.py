"""
Evaluate the Decision Transformer on Breakout by rolling out episodes
conditioned on a target return-to-go.
"""

import os
import numpy as np
import torch
from utils import make_atari_env

@torch.no_grad()
def evaluate_decision_transformer(model, config, target_return=None,
                                   num_episodes=None, render=True):
    """
    Roll out the Decision Transformer in the environment.
    """
    model.eval()
    device = next(model.parameters()).device

    if target_return is None:
        target_return = config.target_return
    if num_episodes is None:
        num_episodes = config.eval_episodes

    # Create environment with rgb_array render mode for video recording
    if render:
        render_mode = "rgb_array"
    else:
        render_mode = None

    env = make_atari_env(config.env_name, config.frame_stack, render_mode=render_mode)

    # Wrap with RecordVideo if render is enabled
    if render:
        import gymnasium as gym
        os.makedirs("videos", exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder="videos/",
            episode_trigger=lambda x: True,  # Record every episode
            name_prefix="dt_breakout"
        )

    episode_returns = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        obs = np.array(obs, dtype=np.float32)  # (C, H, W)

        # Initialize context lists
        states_list = [obs]
        actions_list = [0]  # placeholder, will be overwritten
        rtg_list = [target_return / 100.0]  # normalize same as training
        timestep_list = [0]

        done = False
        truncated = False
        total_reward = 0.0
        t = 0

        while not (done or truncated):
            # Build tensors from context
            states_tensor = torch.tensor(
                np.array(states_list), dtype=torch.float32
            ).unsqueeze(0).to(device)  # (1, T, C, H, W)

            actions_tensor = torch.tensor(
                actions_list, dtype=torch.long
            ).unsqueeze(0).to(device)  # (1, T)

            rtg_tensor = torch.tensor(
                rtg_list, dtype=torch.float32
            ).unsqueeze(0).to(device)  # (1, T)

            timesteps_tensor = torch.tensor(
                timestep_list, dtype=torch.long
            ).unsqueeze(0).to(device)  # (1, T)

            # Get action from model
            action = model.get_action(
                states_tensor, actions_tensor, rtg_tensor, timesteps_tensor
            )

            # Update the last action in context (it was a placeholder)
            actions_list[-1] = action

            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)
            next_obs = np.array(next_obs, dtype=np.float32)
            total_reward += reward
            t += 1

            if not (done or truncated):
                # Append new timestep to context
                new_rtg = rtg_list[-1] - (reward / 100.0)
                states_list.append(next_obs)
                actions_list.append(0)  # placeholder for next action
                rtg_list.append(new_rtg)
                timestep_list.append(min(t, config.max_ep_len - 1))

                # Keep context bounded
                if len(states_list) > config.context_length:
                    states_list = states_list[-config.context_length:]
                    actions_list = actions_list[-config.context_length:]
                    rtg_list = rtg_list[-config.context_length:]
                    timestep_list = timestep_list[-config.context_length:]

        episode_returns.append(total_reward)
        print(f"    Episode {ep + 1}/{num_episodes}: Return = {total_reward:.1f}")

    env.close()
    model.train()

    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    
    if render:
        print(f"\n  Videos saved to: videos/")
    
    return mean_return, std_return