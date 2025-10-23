"""
    TO TEST IT:
    
        python -m envs.frozenlake_custom --size 16 --p 0.95 --max_steps 400 --test_episodes 3
        
        python -m envs.frozenlake_custom --size 16 --p 0.95 --save_map report/sarsa/my_map
"""

from typing import Tuple, Optional
import os
import json
import argparse

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv, generate_random_map
import numpy as np
import textwrap

def make_frozenlake(size: int = 16, p: float = 0.95, is_slippery: bool = True, max_episode_steps: int = 400, seed: Optional[int] = None) -> Tuple[gym.Env, np.ndarray]:
    # creates frozenlake with random map, size x size, u can set prob of safe cells
    if size < 2:
        raise ValueError("Size must be >=2")
    
    # reproducible random map if u want
    if seed is not None:
        rnd_state = np.random.get_state()
        np.random.seed(seed)
        map_list = generate_random_map(size=size, p=p)
        np.random.set_state(rnd_state)
    else:
        map_list = generate_random_map(size=size, p=p)
        
    # map to np array just for utils
    map_desc = np.array([list(row) for row in map_list], dtype="<U1")
    
    # create the env
    env = FrozenLakeEnv(desc=map_list, is_slippery=is_slippery)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    
    return env, map_desc

def map_to_ascii(map_desc: np.ndarray) -> str:
    # make the map look nice as a string for printing
    rows = ["".join(row) for row in map_desc]
    return "\n".join(rows)

def save_map(map_desc: np.ndarray, path: str):
    # saves map as json and txt
    dirpath = os.path.dirname(path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
        
    # json version
    rows = ["".join(row) for row in map_desc]
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        
    # txt version
    with open(path + ".txt", "w", encoding="utf-8") as f:
        f.write(map_to_ascii(map_desc))
        
def _play_random_episode(env: gym.Env, render: bool = False, max_steps: int = 400):
    # play 1 episode randomly, return total reward and if u reach goal
    obs, info = env.reset()
    total_reward = 0.0
    for t in range(max_steps):
        action = env.action_space.sample()  # just pick random
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if render:
            try:
                env.render()
            except Exception:
                pass
        if terminated or truncated:
            return total_reward, bool(reward > 0)
    return total_reward, False

def demo(size: int = 16, p: float = 0.95, is_slippery: bool = True, max_episode_steps: int = 400, seed: Optional[int] = None, test_episodes: int=3, render: bool = False, save_map_to: Optional[str] = None):
    # quick demo, prints map n plays random episodes
    env, map_desc = make_frozenlake(size=size, p=p, is_slippery=is_slippery, max_episode_steps=max_episode_steps, seed=seed)
    
    print("FrozenLake map (S=start, F=frozen, H=hole, G=goal):\n")
    print(map_to_ascii(map_desc))
    if save_map_to:
        save_map(map_desc, save_map_to)
        print(f"\nMap saved at: {save_map_to}.json and .txt")
        
    print("\nTrying random episodes:")
    for i in range(test_episodes):
        total_reward, reached_goal = _play_random_episode(env, render=render, max_steps=max_episode_steps)
        print(f"Episode {i+1}: reward={total_reward:.1f} reached goal={reached_goal}")
        
if __name__ == "__main__":
    # lets u test from terminal, can set size, p, slippery, etc
    parser = argparse.ArgumentParser(description="Generate and test personalized FrozenLake")
    parser.add_argument("--size", type=int, default=16, help="size of the map (size x size)")
    parser.add_argument("--p", type=float, default=0.95, help="probability of safe 'F' cell")
    parser.add_argument("--slippery", action="store_true", help="sif True, environment will be slippery")
    parser.add_argument("--no-slippery", dest="slippery", action="store_false", help="deterministic environment (not slippery)")
    parser.add_argument("--max_steps", type=int, default=400, help="max steps per episode (TimeLimit)")
    parser.add_argument("--seed", type=int, default=None, help="optional seed for reproducibility")
    parser.add_argument("--test_episodes", type=int, default=3, help="episodes of random test")
    parser.add_argument("--render", action="store_true", help="renderize the environment")
    parser.add_argument("--save_map", type=str, default=None, help="base path to save the map")
    
    args = parser.parse_args()
    
    demo(
        size=args.size,
        p=args.p,
        is_slippery=args.slippery,
        max_episode_steps=args.max_steps,
        seed=args.seed,
        test_episodes=args.test_episodes,
        render=args.render,
        save_map_to=args.save_map,
    )
