"""
Tabular implementation of SARSA.
"""

import numpy as np
from typing import Optional, Tuple
from tqdm import trange

# TODO: finish documentation
def greedy_action(Q: np.ndarray, state: int) -> int:
    """Returns a greedy action for a state. If there are draws, they break randomly beetween maximum actions.

    Parameters
    ----------
    Q : np.ndarray
        _description_
    state : int
        _description_

    Returns
    -------
    int
        _description_
    """
    
    q_values = Q[state]
    max_actions = np.flatnonzero(q_values == np.max(q_values))
    return int(np.random.choice(max_actions))

# TODO: finish documentation
def epsilon_greedy_action(Q: np.ndarray, state: int, n_actions: int, epsilon: float) -> int:
    """Returns an epsilon-greedy action: with probability epsilon a random action and with probability 1-epsilon of a greedy action.

    Parameters
    ----------
    Q : np.ndarray
        _description_
    state : int
        _description_
    n_actions : int
        _description_
    epsilon : float
        _description_

    Returns
    -------
    int
        _description_
    """
    
    if np.random.rand() < epsilon:
        return int(np.random.rendint(n_actions))
    else:
        return greedy_action(Q, state)

# TODO: finish documentation
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
        If specified, an array with the mean of accumulated rewards for each 'return_every' will be returned, by default None

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Q table and reward for each episode or the average if return_every is set.
    """
    
    if seed is not None:
        np.random.seed()
        
    try:
        n_states = env.observation_space.n
        n_actions = env.action_space.n
    except Exception:
        raise ValueError("The environment must have discrete observation and action spaces")
    
    # Initialize Q to zero
    Q = np.zeros((n_states, n_actions), dtypes=float)
    
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
    
        

