import os
import argparse

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common import evaluation
from stable_baselines3.common.monitor import Monitor

import stable_worldmodel as swm

def eval_expert(
    env_id: str,
    seed: int = 42,
    eval_episodes: int = 10,
    model_path: str = None,
    save_video: bool = False,
):
    """
    Evaluates a Soft Actor-Critic (SAC) expert policy on a continuous control Fetch environment.
    SAC natively excels at continuous Cartesian robotic control arrays.
    """
    print('===================================================')
    print(f' Evaluating Expert Policy for {env_id}')
    print(f' Setup: SAC | Seed: {seed} | Episodes: {eval_episodes} | Model Path: {model_path}')
    print('===================================================')

    env = gym.make(env_id, render_mode="rgb_array" if save_video else None, resolution=640 if save_video else None)
    env = Monitor(env)

    model = SAC.load(f"{model_path}/final_model", env)

    # episode_rewards, episode_lengths = [], []

    # for episode in range(eval_episodes):
    #     observation, current_reward, current_length, done = env.reset(), 0.0, 0, False
    #     state, episode_start = None, np.ones((1,), dtype=bool)
    #     if save_video:
    #         frames = [env.render()]
    #     while not done:
    #         action, state = model.predict(
    #             observation,  # type: ignore[arg-type]
    #             state=state,
    #             episode_start=episode_start,
    #             deterministic=True,
    #         )
    #         new_observation, reward, done, info = env.step(action)
    #         if save_video:
    #             frames.append(env.render())
    #         current_reward += reward
    #         current_length += 1
    #         observation = new_observation
    #     episode_rewards.append(current_reward)
    #     episode_lengths.append(current_length)
    #     if save_video:


    # mean_reward = np.mean(episode_rewards)
    # std_reward = np.std(episode_rewards)

    if save_video:
        video_dir = f'./policies/{env_id.replace("/", "_")}_expert/video'
        os.makedirs(video_dir, exist_ok=True)
    else:
        video_dir = None    

    if save_video:
        env = gym.wrappers.RecordVideo(env, video_dir, lambda x: True, fps=30)
    mean_reward, std_reward = evaluation.evaluate_policy(model, env, eval_episodes, True, render=save_video)

    print(f'Evaluation complete. Final mean: {mean_reward} | std: {std_reward}$')
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate an RL Expert Policy for Fetch Environments'
    )
    parser.add_argument(
        '--env',
        type=str,
        default='swm/FetchPush-v3',
        help='Target SWM Environment ID',
    )
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')
    parser.add_argument('--eval_episodes', type=int, default=10, help='The number of episodes to evaluate')
    parser.add_argument('--model_path', type=str, default=None, required=True, help='Path of the trained model')
    parser.add_argument('--save_video', action="store_true", help='Save resulting video')

    args = parser.parse_args()

    eval_expert(args.env, args.seed, args.eval_episodes, args.model_path, args.save_video)
