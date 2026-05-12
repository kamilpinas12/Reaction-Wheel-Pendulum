# Reward Function Guide

## Overview

Reward shaping is critical for RL agent learning. This guide explains the different reward wrappers available for your reaction wheel pendulum environment.

## Available Reward Functions

### 1. **SimpleRewardWrapper** ⚡
**Best for:** Quick experiments, debugging, understanding baselines

```yaml
# Config
reward:
  type: "simple"
```

**Formula:**
```
reward = exp(-2.0 * err) - 0.01 * u²
```

**Characteristics:**
- Minimal design (only 2 components)
- Very stable learning signal
- Fastest convergence (but to suboptimal policy)
- Good baseline for comparison

**When to use:**
- Initial debugging
- Hardware testing
- As a baseline for comparison

---

### 2. **BalancedRewardWrapper** ✅ RECOMMENDED
**Best for:** General purpose, good all-around performance

```yaml
# Config
reward:
  type: "balanced"
  params:
    upright_weight: 1.0
    stability_weight: 0.5
    spin_weight: 0.2
    energy_weight: 0.05
    angle_scale: 2.0
```

**Components:**
```
upright_reward = 1.0 * exp(-2.0 * err)
stability_bonus = 0.5 / (1.0 + |θ̇|)     [if err < 0.15]
spin_penalty = 0.2 * θ̇²
energy_penalty = 0.05 * u²

total = upright - spin - energy + stability
```

**Characteristics:**
- 4 balanced components
- Encourages upright position
- Penalizes unnecessary motion
- Energy efficient
- Smooth convergence

**Tuning:**
```yaml
# More aggressive swingup:
upright_weight: 1.5
stability_weight: 0.3

# More energy efficient:
energy_weight: 0.1
spin_weight: 0.3

# Smoother motion:
stability_weight: 1.0
spin_weight: 0.5
```

---

### 3. **ConservativeRewardWrapper** 🛡️
**Best for:** Noisy systems, safety-critical, real-world deployment

```yaml
reward:
  type: "conservative"
  params:
    upright_weight: 1.0
    stability_weight: 1.0
    smooth_weight: 0.3
    energy_weight: 0.1
    phi_limit: 120.0
```

**Components:**
```
upright_reward = 1.0 * exp(-3.0 * err)
stability_bonus = 1.0 * exp(-5.0 * |θ̇|)  [if err < 0.1]
smooth_penalty = 0.3 * (θ̇² + 0.5 * u²)
phi_penalty = 0.01 * max(0, |φ| - 120)²

total = upright + stability - smooth - phi_penalty
```

**Characteristics:**
- Strong emphasis on stability
- Smooth motion encouraged
- Protective phi limit
- Slower convergence, very stable policy
- Good for uncertain parameters

**When to use:**
- Real hardware (safety critical)
- Noisy observations
- Physical constraints matter
- When stability > performance

**Tuning:**
```yaml
# More permissive on wheel rotation:
phi_limit: 140.0

# Slower movements:
smooth_weight: 0.5

# Less strict on staying still:
stability_weight: 0.5
```

---

### 4. **AggressiveRewardWrapper** 🚀
**Best for:** Fast swingup, unconstrained systems, maximizing performance

```yaml
reward:
  type: "aggressive"
  params:
    upright_weight: 2.0
    swingup_bonus: 0.5
    velocity_scale: 0.1
    energy_weight: 0.01
```

**Components:**
```
upright_reward = 2.0 * exp(-1.5 * err)
velocity_bonus = 0.5 * tanh(θ̇ * sign(sin(π - θ)))  [if err < π/2]
energy_penalty = 0.01 * u²

total = upright + velocity_bonus - energy
```

**Characteristics:**
- Rewards movement toward goal
- Fast learning
- Minimal energy penalty
- Can be unstable / oscillatory
- High variance

**When to use:**
- Simulation only (no hardware)
- Need fast convergence
- System has generous limits
- Can tolerate more oscillation

**Tuning:**
```yaml
# More conservative:
upright_weight: 1.5
velocity_scale: 0.2

# Reward even more motion:
swingup_bonus: 1.0

# Reduce oscillations:
energy_weight: 0.05
```

---

### 5. **CurriculumRewardWrapper** 📚
**Best for:** Hard exploration tasks, avoiding local optima

```yaml
reward:
  type: "curriculum"
  params:
    phase_lengths: [0.3, 0.5, 0.2]  # As fraction of total
    total_episodes: 100
```

**Phases:**
1. **Phase 1 (30%):** Simple - just point upright
2. **Phase 2 (50%):** Medium - stay upright, minimize energy
3. **Phase 3 (20%):** Full task - upright + still + efficient

**Characteristics:**
- Starts easy, gradually increases difficulty
- Helps avoid local optima
- Natural learning progression
- Longer training required
- Can be combined with other wrappers

**When to use:**
- Difficult exploration problems
- Want guided learning path
- Have time for training

---

## Comparison Table

| Aspect | Simple | Balanced | Conservative | Aggressive | Curriculum |
|--------|--------|----------|--------------|-----------|-----------|
| **Convergence Speed** | Fast | Medium | Slow | Very Fast | Medium |
| **Stability** | High | Very High | Extreme | Low | Medium |
| **Final Performance** | Low | High | Very High | Medium | High |
| **Oscillation** | Low | Low | Very Low | High | Low |
| **Energy Efficiency** | High | Very High | High | Low | Very High |
| **Exploration** | Low | Medium | Low | High | Medium |
| **Tuning Required** | Low | Low | Medium | Medium | High |

---

## How to Use in Training

### Option 1: Config File (Recommended)

```yaml
# config_a2c.yaml
reward:
  type: "balanced"
  params:
    upright_weight: 1.0
    stability_weight: 0.5
    spin_weight: 0.2
    energy_weight: 0.05
    angle_scale: 2.0

training:
  total_timesteps: 100000
```

Then training automatically uses your chosen reward:
```bash
python scripts/train_a2c.py
```

### Option 2: Programmatic (In Python)

```python
from reac_wheel_sim.reward_wrappers import create_reward_wrapper

# Create environment
env = ReactionWheelEnv()
env = RandomInitialStateWrapper(env, [0.5, 0.5, 0.5])

# Apply reward wrapper
env = create_reward_wrapper(env, reward_type='balanced', 
                           upright_weight=1.0,
                           stability_weight=0.5)

# Train with agent
agent = A2CAgent(env=env, ...)
agent.train(...)
```

### Option 3: Direct Instantiation

```python
from reac_wheel_sim.reward_wrappers import BalancedRewardWrapper

env = ReactionWheelEnv()
env = RandomInitialStateWrapper(env, [0.5, 0.5, 0.5])
env = BalancedRewardWrapper(env, 
                            upright_weight=1.2,
                            stability_weight=0.4)
```

---

## Tuning Strategy

### 1. Start with Balanced
```yaml
reward:
  type: "balanced"  # Safe default
```

### 2. Monitor Training
Watch for:
- ✅ Reward increases smoothly
- ✅ No divergence or collapse
- ⚠️ Converges too slowly → increase `upright_weight`
- ⚠️ Too much spinning → increase `spin_weight`
- ⚠️ Not staying still → increase `stability_weight`

### 3. Adjust Parameters
Based on what you observe:

**Problem: Reward doesn't increase**
```yaml
upright_weight: 1.5  # Increase attraction to upright
energy_weight: 0.02  # Reduce penalty (encourage action)
```

**Problem: Agent spins in circles**
```yaml
spin_weight: 0.5     # Increase spinning penalty
stability_weight: 1.0  # Reward stillness more
```

**Problem: Jerky, aggressive movements**
```yaml
reward_type: "conservative"  # Switch to more stable
smooth_weight: 0.5
```

**Problem: Too slow convergence**
```yaml
upright_weight: 1.5     # Increase goal reward
energy_weight: 0.01     # Reduce energy penalty
```

### 4. Compare with Script
```bash
python scripts/compare_rewards.py
```
This runs all reward functions with random policy and plots:
- Reward distributions
- Statistics comparison
- Recommendations

---

## Advanced: Creating Custom Reward

To create your own reward function:

```python
from reac_wheel_sim.reward_wrappers import RewardWrapper

class MyCustomRewardWrapper(RewardWrapper):
    def __init__(self, env, my_param=1.0):
        super().__init__(env)
        self.my_param = my_param
    
    def reward(self, r):
        theta, theta_dot, phi = self.env.unwrapped.state
        u = self.env.unwrapped.prev_u
        
        # Your custom logic
        err = abs(self._angle_normalize(theta - np.pi))
        custom_reward = self.my_param * np.exp(-2.0 * err)
        
        return float(custom_reward)
```

Then use:
```python
env = MyCustomRewardWrapper(env)
```

---

## Performance Tips

1. **Start Simple**: Begin with `simple` or `balanced`
2. **Monitor Early**: Check loss and reward plots in first 5000 steps
3. **Long Training**: Run at least 100k timesteps for convergence
4. **Compare**: Use `compare_rewards.py` to visualize differences
5. **Iterate**: Small parameter changes usually work better than big ones

---

## FAQ

**Q: Which reward should I use?**
A: Start with `balanced`. It's the default for good reason.

**Q: Why isn't my agent learning?**
A: Usually one of: (1) learning rate too low, (2) upright_weight too low, (3) energy_weight too high.

**Q: How do I measure reward function quality?**
A: Look at final average episode reward and convergence speed. Higher and faster = better.

**Q: Can I use multiple reward types in one training?**
A: Yes, curriculum does this automatically! Or use `NormalizedRewardWrapper` on top.

**Q: Why does my agent spin in circles?**
A: Increase `spin_weight` or switch to `conservative`.

**Q: How long should training take?**
A: Typically 5-15 minutes for 100k timesteps on GPU.
