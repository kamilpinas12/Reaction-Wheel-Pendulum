import gymnasium as gym
import numpy as np


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def _angle_normalize(self, angle):
        """Normalize angle to [-π, π]."""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi
    
    def reward(self, r):
        """Override this in subclasses to compute custom reward."""
        raise NotImplementedError


class SimpleRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, r):
        theta, theta_dot, phi = self.env.unwrapped.state
        height_reward = np.cos(theta - np.pi) 
        normalized_height = (height_reward + 1.0) / 2.0
        return float(normalized_height)


class BalancedRewardWrapper(RewardWrapper): 
    def __init__(self, env, 
                 upright_weight=1.0,
                 stability_weight=0.5, 
                 spin_weight=0.2,
                 energy_weight=0.05):
        super().__init__(env)
        self.upright_weight = upright_weight
        self.stability_weight = stability_weight
        self.spin_weight = spin_weight
        self.energy_weight = energy_weight
    
    def reward(self, r):
        theta, theta_dot, phi = self.env.unwrapped.state
        u = self.env.unwrapped.prev_u
        err = abs(self._angle_normalize(theta - np.pi))

        potential_reward = self.upright_weight * np.cos(err)

        precision_reward = 1.0 * np.exp(-5.0 * err)

        stability_bonus = 0.0
        if err < 0.15:
            stability_bonus = self.stability_weight * np.exp(-abs(theta_dot))

        penalty_mask = 1.0 if err < 1.0 else 0.1
        spin_penalty = self.spin_weight * (theta_dot ** 2) * penalty_mask
        energy_penalty = self.energy_weight * (u ** 2) * penalty_mask
        
        total = potential_reward + precision_reward + stability_bonus - spin_penalty - energy_penalty
        return float(total)

    
class FilipRewardWrapper(RewardWrapper):

    def __init__(self, env):
        super().__init__(env)

    def reward(self, r):
        theta, theta_dot, phi = self.env.unwrapped.state
        u = self.env.unwrapped.prev_u
        err = abs(self._angle_normalize(theta - np.pi))

        # 1. Primary Goal: Stay upright
        # Increased weight to make it more attractive than spinning
        upright_reward = 5.0 * np.exp(-3.0 * err)

        # 2. Phi Penalty: ONLY penalty near the physical limit
        phi_limit = 130.0
        phi_penalty = 0.0
        if abs(phi) > phi_limit:
            phi_penalty = 0.1 * (abs(phi) - phi_limit) ** 2

        # 3. Efficiency: Penalize spinning (The "Anti-Loop" penalty)
        # This is the secret to stopping the multiple circles.
        spinning_penalty = 0.1 * (theta_dot**2)
        energy_penalty = 0.001 * (u ** 2)

        # 4. Stay-Still Bonus: High reward for being at the top AND stopped
        stability_bonus = 0.0
        if err < 0.1:
            stability_bonus = 5.0 / (abs(theta_dot) + 0.1)

        return (
            upright_reward
            + stability_bonus
            - phi_penalty
            - spinning_penalty
            - energy_penalty
        )


# Utility function to create reward wrapper from config
def create_reward_wrapper(env, reward_type='balanced', **kwargs):
    reward_wrappers = {
        'simple': SimpleRewardWrapper,
        'balanced': BalancedRewardWrapper,
        'filip': FilipRewardWrapper
    }
    
    if reward_type not in reward_wrappers:
        raise ValueError(f"Unknown reward type: {reward_type}. Choose from {list(reward_wrappers.keys())}")
    
    wrapper_class = reward_wrappers[reward_type]
    return wrapper_class(env, **kwargs)
