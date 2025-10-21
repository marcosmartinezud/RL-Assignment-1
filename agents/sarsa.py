"""
Tabular implementation of SARSA.
"""

import numpy as np
from typing import Optional, Tuple, Any, Dict
from tqdm import trange
import os
import json

def greedy_action(Q: np.ndarray, state: int) -> int:
    """Chooses the best (greedy) action for the given state, 
    based on the current Q-table. If several actions have the same
    value, it picks one of them randomly.

    Parameters
    ----------
    Q : np.ndarray
        The Q-table with all the values for state-action pairs
    state : int
        The current state number

    Returns
    -------
    int
        The action that has the highest Q value for this state
    """
    
    q_values = Q[state]
    max_actions = np.flatnonzero(q_values == np.max(q_values))
    return int(np.random.choice(max_actions))

def epsilon_greedy_action(Q: np.ndarray, state: int, n_actions: int, epsilon: float) -> int:
    """ Chooses an action using the epsilon-greedy policy.
    This means that most of the time (1 - epsilon) it picks the best action,
    but sometimes (epsilon) it tries a random one to explore.

    Parameters
    ----------
    Q : np.ndarray
        The Q-table used by the agent
    state : int
        The current state number
    n_actions : int
        Number of possible actions in the environment
    epsilon : float
        Probability of exploring (choosing a random action)

    Returns
    -------
    int
        The action chosen by the epsilon-greedy rule.
    """

    if np.random.rand() < epsilon:
        return int(np.random.randint(n_actions))
    else:
        return greedy_action(Q, state)

def train_sarsa(env, n_episodes: int = 2000, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1, min_epsilon: float = 0.01, epsilon_decay: Optional[float] = None, max_steps: Optional[int] = None, seed: Optional[int] = None, verbose: bool = False, return_every: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Train a SARSA agent over env.

    Parameters
    ----------
    env : gym.Env
        Gymnasium environment with descrete observation and action spaces.
    n_episodes : int, optional
        Number of training episodes, by default 2000
    alpha : float, optional
        Learning rate, by default 0.1
    gamma : float, optional
        Discount factor, by default 0.99
    epsilon : float, optional
        Initial value for epsilon for the epsilon-greedy policy, by default 0.1
    min_epsilon : float, optional
        Minimum epsilon value, by default 0.01
    epsilon_decay : Optional[float], optional
        epsilon *= epsilon_decay, by default None
    max_steps : Optional[int], optional
        Limit per episode, by default None
    seed : Optional[int], optional
        Seed for reproducibility, by default None
    verbose : bool, optional
        If true, prints the progress, by default False
    return_every : Optional[int], optional
        If set, returns average rewards every “return_every” episodes, by default None

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The final Q-table and the rewards obtained during training
    """
    
    if seed is not None:
        np.random.seed(seed)
        
    try:
        n_states = env.observation_space.n
        n_actions = env.action_space.n
    except Exception:
        raise ValueError("The environment must have discrete observation and action spaces")
    
    # Initialize Q to zero
    Q = np.zeros((n_states, n_actions), dtype=float)
    
    rewards = np.zeros(n_episodes, dtype=float)
    # If max_steps not specified we use the maximum of the TimeLimit wrapper if it exists
    if max_steps is None:
        max_steps = getattr(env, "spec", None)
        # reasonable fallback
        if max_steps is None:
            max_steps = 1000
            
    iterator = trange(n_episodes, desc="Training SARSA") if verbose else range(n_episodes)
    for ep in iterator:
        # initialize episode
        reset_kwargs = {}
        if seed is not None:
            reset_kwargs["seed"] = seed + ep
        obs, info = env.reset(**reset_kwargs)
        state = int(obs)
    
        # choose initial action using epsilon-greedy
        action = epsilon_greedy_action(Q, state, n_actions, epsilon)
    
        total_reward = 0.0
        for t in range(max_steps):
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = int(next_obs)
            done = bool(terminated or truncated)
            
            # choose next aciton using the same epsilon-greedy policy
            next_action = epsilon_greedy_action(Q, next_state, n_actions, epsilon)
            
            # TD target and update
            td_target = reward + (0.0 if done else gamma * Q[next_state, next_action])
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error
            
            total_reward += float(reward)
            
            # advance
            state = next_state
            action = next_action
        
            if done:
                break
        
        rewards[ep] = total_reward
    
        # update epsilon if decay applies
        if epsilon_decay is not None:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            
        # optional progress
        if verbose and (ep % 100 == 0):
            iterator.set_postfix({"epsiode": ep, "ep_reward": total_reward, "epsilon": epsilon})
        
    # if user asks for measurements each X episodes
    if return_every is not None and return_every > 0:
        n_chunks = int(np.ceil(len(rewards) / return_every))
        rewards_avg = np.array([rewards[i * return_every:(i+1)*return_every].mean() for i in range(n_chunks)])
        return Q, rewards_avg
    
    return Q, rewards
    
def evaluate_policy(Q: np.ndarray, env, n_episodes: int = 100, max_steps: Optional[int] = None, render: bool = False, seed: Optional[int] = None) -> Dict[str, Any]:
    """tests the learned Q-table by running episodes without exploration
    (the agent only takes the best actions).

    Parameters
    ----------
    Q : np.ndarray
        The Q-table to evaluate
    env : _type_
        The environment to test in
    n_episodes : int, optional
        Number of test episodes, by default 100
    max_steps : Optional[int], optional
        Max number of steps per episode, by default None
    render : bool, optional
        If True, shows the environment visually, by default False
    seed : Optional[int], optional
        Random seed for reproducibility, by default None

    Returns
    -------
    Dict[str, Any]
        Dictionary with the average reward, success rate, and
        a list of all rewards per episode.
    """
    
    if seed is not None:
        np.random.seed(seed)

    if max_steps is None:
        max_steps = getattr(env, "spec", None) or 1000

    n_states, n_actions = Q.shape
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
            action = greedy_action(Q, state)
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

def save_q(Q: np.ndarray, path: str):
    """Save the Q-table in a .npy and .json file

    Parameters
    ----------
    Q : np.ndarray
        Q-table to save.
    path : str
        The path (without extension) where the files will be saved
    """
    dirpath = os.path.dirname(path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    np.save(path + ".npy", Q)
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(Q.tolist(), f, indent=2)
        
def load_q(path: str) -> np.ndarray:
    """Load the Q-table from the .npy file

    Parameters
    ----------
    path : str
        The path (without extension) to the file

    Returns
    -------
    np.ndarray
        The Q-table loaded from disk
    """
        
    return np.load(path + ".npy")

