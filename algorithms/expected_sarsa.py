"""
    TO RUN IT: python -m algorithms.expected_sarsa
"""

import numpy as np
from typing import Optional
from tqdm import trange
import os
import json

# picks the best action (the one with highest value in Q)
def greedy_action(Q: np.ndarray, state: int) -> int:
    q_values = Q[state]
    max_actions = np.flatnonzero(q_values == np.max(q_values))
    return int(np.random.choice(max_actions))

# epsilon greedy policy: sometimes explore, sometimes exploit
def epsilon_greedy_action(Q: np.ndarray, state: int, n_actions: int, epsilon: float) -> int:
    if np.random.rand() < epsilon:
        return int(np.random.randint(n_actions))
    else:
        return greedy_action(Q, state)

# expected q value under epsilon-greedy policy
def _expected_q(Q: np.ndarray, state: int, n_actions: int, epsilon: float) -> float:
    q_values = Q[state]
    max_value = np.max(q_values)
    greedy_mask = (q_values == max_value)
    n_greedy = int(np.sum(greedy_mask))
    if n_greedy == 0:
        # avoid division by zero if something weird happens
        n_greedy = 1
        greedy_mask = np.ones_like(q_values, dtype=bool)

    greedy_prob = (1 - epsilon) / n_greedy + epsilon / n_actions
    non_greedy_prob = epsilon / n_actions

    probs = np.full(n_actions, non_greedy_prob, dtype=float)
    probs[greedy_mask] = greedy_prob

    return float(np.dot(probs, q_values))

# main Expected-SARSA training function
def train_expected_sarsa(env, n_episodes: int = 2000, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1,
                         min_epsilon: float = 0.01, epsilon_decay: Optional[float] = None, max_steps: Optional[int] = None,
                         seed: Optional[int] = None, verbose: bool = False, return_every: Optional[int] = None):
    
    if seed is not None:
        np.random.seed(seed)  # reproducible

    try:
        n_states = env.observation_space.n
        n_actions = env.action_space.n
    except Exception:
        raise ValueError("env must have discrete observation and action spaces")

    Q = np.zeros((n_states, n_actions), dtype=float)
    rewards = np.zeros(n_episodes, dtype=float)
    max_steps = max_steps or 1000

    iterator = trange(n_episodes, desc="Training Expected SARSA") if verbose else range(n_episodes)
    for ep in iterator:
        reset_kwargs = {}
        if seed is not None:
            reset_kwargs["seed"] = seed + ep
        obs, info = env.reset(**reset_kwargs)
        state = int(obs)
        total_reward = 0.0

        for t in range(max_steps):
            action = epsilon_greedy_action(Q, state, n_actions, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = int(next_obs)
            done = bool(terminated or truncated)

            expected_next = 0.0 if done else _expected_q(Q, next_state, n_actions, epsilon)
            td_target = reward + gamma * expected_next
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            total_reward += float(reward)
            state = next_state

            if done:
                break

        rewards[ep] = total_reward

        # epsilon decay over time
        if epsilon_decay is not None:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if verbose and (ep % 100 == 0):
            iterator.set_postfix({"episode": ep, "ep_reward": total_reward, "epsilon": epsilon})

    # if user wants avg every X episodes
    if return_every is not None and return_every > 0:
        n_chunks = int(np.ceil(len(rewards) / return_every))
        rewards_avg = np.array([rewards[i * return_every:(i + 1) * return_every].mean() for i in range(n_chunks)])
        return Q, rewards_avg

    return Q, rewards

# test policy (no exploration)
def evaluate_policy(Q: np.ndarray, env, n_episodes: int = 100, max_steps: Optional[int] = None,
                    render: bool = False, seed: Optional[int] = None):
    
    if seed is not None:
        np.random.seed(seed)

    max_steps = max_steps or 1000
    rewards = np.zeros(n_episodes, dtype=float)
    successes = 0

    for ep in range(n_episodes):
        reset_kwargs = {}
        if seed is not None:
            reset_kwargs["seed"] = seed + 10000 + ep
        obs, info = env.reset(**reset_kwargs)
        state = int(obs)
        total_reward = 0.0

        for t in range(max_steps):
            action = greedy_action(Q, state)  # always greedy now
            obs, reward, terminated, truncated, info = env.step(action)
            state = int(obs)
            total_reward += float(reward)

            if render:
                try:
                    env.render()
                except Exception:
                    pass

            if terminated or truncated:
                break

        rewards[ep] = total_reward
        if total_reward > 0:
            successes += 1

    return {"average_reward": float(rewards.mean()), "success_rate": float(successes / n_episodes), "rewards": rewards}

# save Q-table to npy and json (for checking)
def save_q(Q: np.ndarray, path: str):
    dirpath = os.path.dirname(path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    np.save(path + ".npy", Q)
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(Q.tolist(), f, indent=2)

# load Q-table from file
def load_q(path: str) -> np.ndarray:
    return np.load(path + ".npy")


if __name__ == "__main__":
    try:
        import gymnasium as gym
        from envs.frozenlake_custom import make_frozenlake

        env, _ = make_frozenlake(size=8, p=0.95, is_slippery=True, max_episode_steps=200, seed=42)
        Q, rewards = train_expected_sarsa(env, n_episodes=100, alpha=0.1, gamma=0.99, epsilon=0.2, seed=42, verbose=True)
        print("Finished quick run. avg reward (last 10 eps):", float(rewards[-10:].mean()))
    except Exception as e:
        print("Quick self-test skipped (missing deps).", e)
