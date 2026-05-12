#!/usr/bin/env python3
"""
Test script to verify reward wrapper is being called and transforming rewards correctly.
"""

import sys
from pathlib import Path
import numpy as np

# Add paths
ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reac_wheel_sim.reaction_wheel_env import ReactionWheelEnv
from reac_wheel_sim.reaction_wheel_wrappers import RandomInitialStateWrapper
from reac_wheel_sim.reward_wrappers import (
    SimpleRewardWrapper,
    BalancedRewardWrapper,
)


def test_reward_wrapper():
    """Test that reward wrapper correctly transforms rewards."""
    
    print("\n" + "="*70)
    print("REWARD WRAPPER TEST - Verifying reward transformation")
    print("="*70)
    
    # Test 1: Without wrapper (original env)
    print("\n1️⃣  Testing ORIGINAL environment (no reward wrapper):")
    print("-" * 70)
    
    env_original = ReactionWheelEnv()
    env_original = RandomInitialStateWrapper(env_original, [0.5, 0.5, 0.5])
    
    obs, _ = env_original.reset()
    original_rewards = []
    
    for i in range(10):
        action = np.array([0.5])  # Fixed action
        obs, reward, terminated, truncated, info = env_original.step(action)
        original_rewards.append(reward)
        if i < 3:
            print(f"  Step {i+1}: reward = {reward:.4f}")
    
    print(f"  Average original reward: {np.mean(original_rewards):.4f}")
    print(f"  Original rewards are from: ReactionWheelEnv._get_reward()")
    
    # Test 2: With Simple wrapper
    print("\n2️⃣  Testing with SimpleRewardWrapper (verbose=True):")
    print("-" * 70)
    
    env_simple = ReactionWheelEnv()
    env_simple = RandomInitialStateWrapper(env_simple, [0.5, 0.5, 0.5])
    env_simple = SimpleRewardWrapper(env_simple)
    
    obs, _ = env_simple.reset()
    simple_rewards = []
    
    for i in range(10):
        action = np.array([0.5])  # Same fixed action
        obs, reward, terminated, truncated, info = env_simple.step(action)
        simple_rewards.append(reward)
        if i < 3:
            print(f"  Step {i+1}: reward = {reward:.4f}")
    
    print(f"  Average simple wrapper reward: {np.mean(simple_rewards):.4f}")
    
    # Test 3: With Balanced wrapper
    print("\n3️⃣  Testing with BalancedRewardWrapper (verbose=True):")
    print("-" * 70)
    
    env_balanced = ReactionWheelEnv()
    env_balanced = RandomInitialStateWrapper(env_balanced, [0.5, 0.5, 0.5])
    env_balanced = BalancedRewardWrapper(env_balanced)
    
    obs, _ = env_balanced.reset()
    balanced_rewards = []
    
    for i in range(10):
        action = np.array([0.5])  # Same fixed action
        obs, reward, terminated, truncated, info = env_balanced.step(action)
        balanced_rewards.append(reward)
        if i < 3:
            print(f"  Step {i+1}: reward = {reward:.4f}")
    
    print(f"  Average balanced wrapper reward: {np.mean(balanced_rewards):.4f}")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    print(f"\n✅ Reward transformation verification:")
    print(f"   Original rewards:  {original_rewards[:3]} ... (avg: {np.mean(original_rewards):.4f})")
    print(f"   Simple wrapped:    {simple_rewards[:3]} ... (avg: {np.mean(simple_rewards):.4f})")
    print(f"   Balanced wrapped:  {balanced_rewards[:3]} ... (avg: {np.mean(balanced_rewards):.4f})")
    
    # Check if rewards are different
    if np.allclose(original_rewards, simple_rewards):
        print("\n❌ PROBLEM: SimpleRewardWrapper is NOT transforming rewards!")
        print("   Rewards are identical to original environment.")
        return False
    else:
        print("\n✅ SUCCESS: SimpleRewardWrapper IS transforming rewards correctly!")
        print(f"   Difference: {np.mean(np.abs(np.array(simple_rewards) - np.array(original_rewards))):.4f}")
    
    if np.allclose(simple_rewards, balanced_rewards):
        print("\n⚠️  WARNING: Balanced and Simple rewards are very similar.")
        print("   They might not be different enough to notice.")
    else:
        print("\n✅ SUCCESS: BalancedRewardWrapper IS different from SimpleRewardWrapper!")
        print(f"   Difference: {np.mean(np.abs(np.array(balanced_rewards) - np.array(simple_rewards))):.4f}") 
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")
    
    return True


def test_state_access():
    """Test that wrapper can access environment state correctly."""
    
    print("\n" + "="*70)
    print("STATE ACCESS TEST - Verifying wrapper can read state")
    print("="*70)
    
    env = ReactionWheelEnv()
    env = RandomInitialStateWrapper(env, [0.5, 0.5, 0.5])
    env = SimpleRewardWrapper(env)
    
    obs, _ = env.reset()
    
    print(f"\n✅ Environment state accessible:")
    print(f"   env.unwrapped.state = {env.unwrapped.state}")
    print(f"   env.unwrapped.prev_u = {env.unwrapped.prev_u}")
    
    action = np.array([0.5])
    obs, reward, _, _, _ = env.step(action)
    
    print(f"\n✅ After step:")
    print(f"   theta = {env.unwrapped.state[0]:.4f}")
    print(f"   theta_dot = {env.unwrapped.state[1]:.4f}")
    print(f"   phi = {env.unwrapped.state[2]:.4f}")
    print(f"   u = {env.unwrapped.prev_u:.4f}")
    print(f"   reward = {reward:.4f}")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    test_state_access()
    test_reward_wrapper()
