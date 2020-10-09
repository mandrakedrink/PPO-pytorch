import gym
import numpy as np

class PendulumActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action

    
class AcrobotActionWrapper(gym.ActionWrapper):
    """Change the action range (0, 2) to (-1, -1)."""

    def action(self, action: np.ndarray) -> np.ndarray:
        # modify act
        remap = {0:-1, 1:0, 2:1}
        action = remap[action]
        return action
    

class BipedalWalkerRewardWrapper(gym.RewardWrapper):
    """
    Rescale the negative rewards from -100 to -1.
    https://github.com/jet-black/ppo-lstm-parallel/blob/master/reward.py
    """
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return max(-1.0, reward)
