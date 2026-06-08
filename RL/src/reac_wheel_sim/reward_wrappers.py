import gymnasium as gym
import numpy as np
import inspect


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
                 energy_weight=0.05,
                 wheel_vel_weight=0.5):
        super().__init__(env)
        self.upright_weight = upright_weight
        self.stability_weight = stability_weight
        self.spin_weight = spin_weight
        self.energy_weight = energy_weight
        self.wheel_vel_weight = wheel_vel_weight
    
    def reward(self, r):
        pend_pos, pend_vel, wheel_vel = self.env.unwrapped.state
        u = self.env.unwrapped.prev_u
        err = abs(self._angle_normalize(pend_pos - np.pi))

        potential_reward = self.upright_weight * (np.cos(err) + 1.0) / 2.0

        stability_bonus = 0.0
        if err < 0.5:
            stability_bonus = self.stability_weight * (np.exp(-abs(pend_vel)))
        
        spin_penalty = self.spin_weight * ((pend_vel / 2.0)**2) 
        energy_penalty = self.energy_weight * (u ** 2)
        wheel_penalty = self.wheel_vel_weight * ((wheel_vel/ 300.0) ** 2)
        
        total = potential_reward + stability_bonus - spin_penalty - energy_penalty - wheel_penalty
        return float(total)

    
class FilipRewardWrapper(RewardWrapper):

    def __init__(
        self,
        env,
        upright_gain=5.0,
        upright_exp=3.0,
        phi_limit=130.0,
        phi_penalty_weight=0.1,
        spin_penalty_weight=0.1,
        energy_penalty_weight=0.001,
        stability_err_threshold=0.1,
        stability_bonus_gain=5.0,
        stability_bonus_offset=0.1,
    ):
        super().__init__(env)
        self.upright_gain = float(upright_gain)
        self.upright_exp = float(upright_exp)
        self.phi_limit = float(phi_limit)
        self.phi_penalty_weight = float(phi_penalty_weight)
        self.spin_penalty_weight = float(spin_penalty_weight)
        self.energy_penalty_weight = float(energy_penalty_weight)
        self.stability_err_threshold = float(stability_err_threshold)
        self.stability_bonus_gain = float(stability_bonus_gain)
        self.stability_bonus_offset = float(stability_bonus_offset)

    def reward(self, r):
        theta, theta_dot, phi = self.env.unwrapped.state
        u = self.env.unwrapped.prev_u
        err = abs(self._angle_normalize(theta - np.pi))

        upright_reward = self.upright_gain * np.exp(-self.upright_exp * err)

        phi_penalty = 0.0
        if abs(phi) > self.phi_limit:
            phi_penalty = self.phi_penalty_weight * (abs(phi) - self.phi_limit) ** 2

        spinning_penalty = self.spin_penalty_weight * (theta_dot**2)
        energy_penalty = self.energy_penalty_weight * (u ** 2)

        stability_bonus = 0.0
        if err < self.stability_err_threshold:
            stability_bonus = self.stability_bonus_gain / (abs(theta_dot) + self.stability_bonus_offset)

        return float(
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
    # Allow passing reward params from config without crashing when unknown keys are present.
    signature = inspect.signature(wrapper_class.__init__)
    valid_keys = {k for k in signature.parameters.keys() if k not in {"self", "env"}}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    return wrapper_class(env, **filtered_kwargs)
