import gymnasium as gym
import numpy as np
from stable_worldmodel.policy import BasePolicy

class ExpertPolicy(BasePolicy):
    def __init__(
        self,
        ckpt_path: str,
        device: str = 'cpu',
        env: gym.Env = None,
        obs_keys: list[str] = None,
        **kwargs,
    ):
        """
        Expert Policy (RL) for Gymnasium Robotics Fetch environments.
        Policies have been trained using Soft Actor-Critic (SAC) algorithm from stable_baselines3.

        Args:
            ckpt_path (str): Path to the stable_baselines3 .zip file of the trained policy.
            device (str): Device to load the model on, e.g., 'cpu' or 'cuda'.
            env (gym.Env): Gymnasium environment to infer action space and observation space from.
            obs_keys (list[str]): List of observation keys to use.
        """
        super().__init__(**kwargs)

        try:
            import stable_baselines3 as sb3
        except ImportError:
            raise ImportError(
                'stable_baselines3 is required to use the ExpertPolicy. '
                "Please install it via 'uv add stable-baselines3'."
            )

        self.model = sb3.SAC.load(ckpt_path, device=device, env=env)
        self.type = 'expert'
        self.states = None
        self.obs_keys = obs_keys

    def set_env(self, env):
        self.env = env

    def get_action(self, info_dict, **kwargs):
        assert all(key in info_dict.keys() for key in self.obs_keys), (
            'One or more observation keys missing in info_dict'
        )

        obs_dict = {}
        for obs_key in self.obs_keys:
            obs_dict[obs_key] = info_dict[obs_key].squeeze()

        actions, self.states = self.model.predict(
            obs_dict,
            state=self.states,
            episode_start=~np.logical_or(info_dict['terminated'], info_dict['truncated']).squeeze(),
            deterministic=True,
        )
        return np.clip(actions, -1.0, 1.0).astype(np.float32)
