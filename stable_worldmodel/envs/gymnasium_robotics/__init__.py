import gymnasium as gym
import gymnasium_robotics

from .fetch import FetchWrapper
from .expert_policy import ExpertPolicy

gym.register_envs(gymnasium_robotics)

__all__ = ['FetchWrapper', 'ExpertPolicy']
