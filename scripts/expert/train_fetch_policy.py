import os
import argparse
import datetime
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
except ImportError:
    wandb = None
    WandbCallback = None

import stable_worldmodel as swm

# Define default architectures and hyperparameters

ARCH_SMALL = {'net_arch': [256, 256]}
ARCH_MEDIUM = {'net_arch': [400, 300]}
ARCH_LARGE = {'net_arch': [1024, 1024]}

DEFAULT_CFG = {
    'batch_size': 256,
    'policy_kwargs': ARCH_SMALL,
    'learning_starts': 10000,
}

QUADRUPED_CFG = {
    'batch_size': 1024,
    'gradient_steps': 1,
    'learning_starts': 10000,
    'policy_kwargs': ARCH_MEDIUM,
    'tau': 0.005,
}

WALKER_CFG = {
    'batch_size': 1024,
    'gradient_steps': 2,
    'train_freq': 1,
    'policy_kwargs': ARCH_MEDIUM,
    'learning_starts': 10000,
    'tau': 0.005,
}

HUMANOID_CFG = {
    'batch_size': 1024,
    'gradient_steps': 2,
    'policy_kwargs': ARCH_LARGE,
    'learning_starts': 25000,
}

# Registry mapping domains and tasks to their SAC hyperparameters
PARAMS_REGISTRY = {
    'pendulum': {
        'swingup': {
            **DEFAULT_CFG,
            'gradient_steps': 2,
            'batch_size': 1024,
            'learning_starts': 25000,
            'policy_kwargs': ARCH_MEDIUM,
            'total_timesteps': 750_000,
        }
    },
    'ballincup': {'default': {**DEFAULT_CFG, 'total_timesteps': 750_000}},
    'cartpole': {'default': {**DEFAULT_CFG, 'total_timesteps': 750_000}},
    'quadruped': {
        'walk': {**QUADRUPED_CFG, 'total_timesteps': 2_500_000},
        'run': {**QUADRUPED_CFG, 'total_timesteps': 3_500_000},
    },
    'cheetah': {
        'run': {**DEFAULT_CFG, 'total_timesteps': 750_000},
        'run-backward': {**DEFAULT_CFG, 'total_timesteps': 750_000},
        'run-front': {**DEFAULT_CFG, 'total_timesteps': 750_000},
        'run-back': {**DEFAULT_CFG, 'total_timesteps': 750_000},
        'stand-front': {**DEFAULT_CFG, 'total_timesteps': 1_000_000},
        'stand-back': {**DEFAULT_CFG, 'total_timesteps': 1_000_000},
        'lie-down': {**DEFAULT_CFG, 'total_timesteps': 1_000_000},
        'jump': {**DEFAULT_CFG, 'total_timesteps': 1_500_000},
        'legs-up': {**DEFAULT_CFG, 'total_timesteps': 1_500_000},
        'flip': {**DEFAULT_CFG, 'total_timesteps': 1_500_000},
        'flip-backward': {**DEFAULT_CFG, 'total_timesteps': 1_500_000},
    },
    'reacher': {
        'easy': {
            **DEFAULT_CFG,
            'total_timesteps': 750_000,
            'learning_starts': 5000,
        },
        'hard': {
            **DEFAULT_CFG,
            'total_timesteps': 1_000_000,
            'learning_starts': 5000,
        },
    },
    'walker': {
        'stand': {**WALKER_CFG, 'total_timesteps': 1_000_000},
        'walk': {**WALKER_CFG, 'total_timesteps': 1_000_000},
        'run': {**WALKER_CFG, 'total_timesteps': 1_500_000},
        'walk-backward': {**WALKER_CFG, 'total_timesteps': 1_500_000},
        'lie_down': {**WALKER_CFG, 'total_timesteps': 1_500_000},
        'flip': {**WALKER_CFG, 'total_timesteps': 2_500_000},
        'arabesque': {**WALKER_CFG, 'total_timesteps': 2_500_000},
        'legs_up': {**WALKER_CFG, 'total_timesteps': 2_500_000},
    },
    'hopper': {
        'stand': {
            **DEFAULT_CFG,
            'batch_size': 1024,
            'gradient_steps': 2,
            'tau': 0.005,
            'total_timesteps': 2_000_000,
        },
        'hop': {
            **DEFAULT_CFG,
            'batch_size': 1024,
            'gradient_steps': 2,
            'tau': 0.005,
            'total_timesteps': 2_500_000,
        },
        'hop-backward': {
            **DEFAULT_CFG,
            'batch_size': 1024,
            'gradient_steps': 2,
            'tau': 0.005,
            'total_timesteps': 4_000_000,
        },
        'flip': {
            **DEFAULT_CFG,
            'batch_size': 1024,
            'gradient_steps': 2,
            'tau': 0.005,
            'total_timesteps': 4_000_000,
        },
        'flip-backward': {
            **DEFAULT_CFG,
            'batch_size': 1024,
            'gradient_steps': 2,
            'tau': 0.005,
            'total_timesteps': 4_000_000,
        },
    },
    'finger': {
        'spin': {**DEFAULT_CFG, 'total_timesteps': 750_000},
        'turn_easy': {**DEFAULT_CFG, 'total_timesteps': 1_000_000},
        'turn_hard': {**DEFAULT_CFG, 'total_timesteps': 1_500_000},
    },
    'humanoid': {
        'stand': {**HUMANOID_CFG, 'total_timesteps': 5_000_000},
        'walk': {**HUMANOID_CFG, 'total_timesteps': 5_000_000},
        'run': {**HUMANOID_CFG, 'total_timesteps': 5_000_000},
    },
}

def train_expert(
    env_id: str,
    total_timesteps: int,
    seed: int = 42,
    track: bool = False,
    project_name: str = 'stable-worldmodel',
):
    """
    Trains a Soft Actor-Critic (SAC) expert policy on a continuous control Fetch environment.
    SAC natively excels at continuous Cartesian robotic control arrays.
    """
    print('===================================================')
    print(f' Training Expert Policy for {env_id}')
    print(f' Setup: SAC | {total_timesteps} Timesteps | Seed: {seed}')
    print('===================================================')

    env = Monitor(gym.make(env_id))
    eval_env = Monitor(gym.make(env_id))

    model = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        seed=seed,
        tensorboard_log=f'./logs/tensorboard/{env_id.replace("/", "_")}_sac/{datetime.datetime.now().strftime("%d%m%y/%H%M%S")}',
        learning_rate=3e-4,
        batch_size=2048,
        gradient_steps=2,
        train_freq=1,
        policy_kwargs={"net_arch": [512, 512], "n_critics": 2},
        learning_starts=10000,
        tau=0.005,
    )

    save_path = f'./policies/{env_id.replace("/", "_")}_expert/{datetime.datetime.now().strftime("%d%m%y/%H%M%S")}'
    os.makedirs(save_path, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    callbacks = [eval_callback]

    if track and wandb is None:
        raise ImportError(
            'wandb is required for tracking. Install it with: pip install wandb'
        )

    if track:
        wandb.init(
            project=project_name,
            name=f'SAC_{env_id.replace("/", "_")}',
            config={
                'env': env_id,
                'algo': 'SAC',
                'seed': seed,
                'timesteps': total_timesteps,
            },
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        wandb_callback = WandbCallback(
            model_save_path=save_path, model_save_freq=5000, verbose=2
        )
        callbacks.append(wandb_callback)

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True,
    )

    model.save(f'{save_path}/final_model')

    if track:
        wandb.finish()

    print(f'Training complete. Models saved to {save_path}')
    env.close()
    eval_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train an RL Expert Policy for Fetch Environments'
    )
    parser.add_argument(
        '--env',
        type=str,
        default='swm/FetchPush-v3',
        help='Target SWM Environment ID',
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=100000,
        help='Total environment steps to execute',
    )
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')
    parser.add_argument(
        '--track',
        action='store_true',
        help='Log training metrics natively to Weights & Biases',
    )
    parser.add_argument(
        '--project',
        type=str,
        default='stable-worldmodel',
        help='WandB Cloud project name',
    )

    args = parser.parse_args()

    train_expert(args.env, args.timesteps, args.seed, args.track, args.project)
