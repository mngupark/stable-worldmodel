import numpy as np
from stable_worldmodel.policy import BasePolicy

class ExpertPolicy(BasePolicy):
    def __init__(
        self,
        ckpt_path: str,
        device: str = 'cpu',
        **kwargs,
    ):
        """
        Expert Policy (RL) for Gymnasium Robotics Fetch environments.
        Policies have been trained using Soft Actor-Critic (SAC) algorithm from stable_baselines3.

        Args:
            ckpt_path (str): Path to the stable_baselines3 .zip file of the trained policy.
            device (str): Device to load the model on, e.g., 'cpu' or 'cuda'.
        """
        super().__init__(**kwargs)

        try:
            import stable_baselines3 as sb3
        except ImportError:
            raise ImportError(
                'stable_baselines3 is required to use the ExpertPolicy. '
                "Please install it via 'uv add stable-baselines3'."
            )

        self.model = sb3.SAC.load(ckpt_path, device=device)
        self.type = 'expert'

    def set_env(self, env):
        self.env = env

    def get_action(self, info_dict, **kwargs):
        assert 'observation' in info_dict, (
            'Observation key missing in info_dict'
        )

        obs = info_dict['observation'].squeeze()

        if obs.ndim == 1:
            obs = obs[None, :]

        if len(obs.shape) != 2:
            raise ValueError(
                f'Expected observation shape (num_envs, obs_dim), got {obs.shape}'
            )

        actions, _ = self.model.predict(obs, deterministic=True)
        return np.clip(actions, -1.0, 1.0).astype(np.float32)
