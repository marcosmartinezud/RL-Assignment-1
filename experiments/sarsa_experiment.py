import gymnasium as gym
import optuna
from algorithms.sarsa import train_sarsa, evaluate_policy, save_q
from envs.frozenlake_custom import make_frozenlake
import numpy as np
import os
import json

# global env settings
MAP_SIZE = 16
P_SAFE = 0.95
MAX_EPISODE_STEPS = 400
SEED = 123  # u can change this if u want

def objective(trial):
    # Optuna gonna try different hyperparams
    alpha = trial.suggest_float("alpha", 0.01, 0.5)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    epsilon = trial.suggest_float("epsilon", 0.05, 0.3)
    min_epsilon = trial.suggest_float("min_epsilon", 0.001, 0.05)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.95, 0.999)

    # make the custom frozenlake
    env, map_desc = make_frozenlake(
        size=MAP_SIZE,
        p=P_SAFE,
        is_slippery=True,
        max_episode_steps=MAX_EPISODE_STEPS,
        seed=SEED
    )

    # train sarsa
    Q, _ = train_sarsa(
        env,
        n_episodes=2000,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        min_epsilon=min_epsilon,
        epsilon_decay=epsilon_decay,
        max_steps=MAX_EPISODE_STEPS,
        seed=SEED,
        verbose=False
    )

    # test without exploration
    eval_result = evaluate_policy(
        Q, env,
        n_episodes=100,
        max_steps=MAX_EPISODE_STEPS,
        render=False,
        seed=SEED
    )
    return eval_result["average_reward"]

def main():
    # setup Optuna study (creates db inside /data)
    os.makedirs("data", exist_ok=True)
    os.makedirs("report/sarsa", exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name="sarsa_frozenlake",
        storage="sqlite:///data/sarsa/optuna_sarsa.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=50)  # can do more if u want

    print("best hyperparams found:")
    print(study.best_params)

    # save best params to json inside report/sarsa/
    with open("report/sarsa/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

    # train again with best params (final version)
    env, _ = make_frozenlake(
        size=MAP_SIZE,
        p=P_SAFE,
        is_slippery=True,
        max_episode_steps=MAX_EPISODE_STEPS,
        seed=SEED
    )

    best_params = study.best_params
    Q, _ = train_sarsa(
        env,
        n_episodes=5000,
        alpha=best_params["alpha"],
        gamma=best_params["gamma"],
        epsilon=best_params["epsilon"],
        min_epsilon=best_params["min_epsilon"],
        epsilon_decay=best_params["epsilon_decay"],
        max_steps=MAX_EPISODE_STEPS,
        seed=SEED,
        verbose=True
    )

    # save Q-table in both formats (.npy and .json)
    np.save("report/sarsa/best_q.npy", Q)
    with open("report/sarsa/best_q.json", "w") as f:
        json.dump(Q.tolist(), f, indent=2)

    # also save it using the helper just in case
    save_q(Q, "report/sarsa/best_q")

if __name__ == "__main__":
    main()
