"""
Tabular implementation of SARSA.
"""

import numpy as np
from typing import Optional, Tuple

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
    """_summary_

    Parameters
    ----------
    env : _type_
        _description_
    n_episodes : int, optional
        _description_, by default 2000
    alpha : float, optional
        _description_, by default 0.1
    gamma : float, optional
        _description_, by default 0.99
    epsilon : float, optional
        _description_, by default 0.1
    min_epsilon : float, optional
        _description_, by default 0.01
    epsilon_decay : Optional[float], optional
        _description_, by default None
    max_steps : Optional[int], optional
        _description_, by default None
    seed : Optional[int], optional
        _description_, by default None
    verbose : bool, optional
        _description_, by default False
    return_every : Optional[int], optional
        _description_, by default None

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        _description_
    """
    None

