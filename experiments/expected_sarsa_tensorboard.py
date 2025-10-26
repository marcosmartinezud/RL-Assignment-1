"""
    TO RUN IT: python -m experiments.expected_sarsa_tensorboard
    Logs rewards, moving average, epsilon, and success metrics to TensorBoard
"""

import os
import numpy as np
import optuna
from torch.utils.tensorboard import SummaryWriter
from algorithms.expected_sarsa import train_expected_sarsa, evaluate_policy, _expected_q, epsilon_greedy_action
from envs.frozenlake_custom import make_frozenlake

# Global environment setup
MAP_SIZE = 16
P_SAFE = 0.95
MAX_EPISODE_STEPS = 400
SEED = 123

# Training parameters
LONG_TRAIN_EPISODES = 20000
EVAL_INTERVAL = 500
MOVING_AVG_WINDOW = 100


def main():
    study = optuna.load_study(
        study_name="expected_sarsa_frozenlake",
        storage="sqlite:///data/expected_sarsa/optuna_expected_sarsa.db"
    )

    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else 0, reverse=True)[:3]
    print("Top 3 hyperparameter sets:")
    for i, trial in enumerate(top_trials, start=1):
        print(f"Trial {i} params:", trial.params)

    base_logdir = "runs/expected_sarsa"
    os.makedirs(base_logdir, exist_ok=True)
    os.makedirs("report/expected_sarsa", exist_ok=True)

    env, _ = make_frozenlake(
        size=MAP_SIZE,
        p=P_SAFE,
        is_slippery=True,
        max_episode_steps=MAX_EPISODE_STEPS,
        seed=SEED
    )

    for i, trial in enumerate(top_trials, start=1):
        params = trial.params
        writer = SummaryWriter(log_dir=os.path.join(base_logdir, f"best_{i}"))

        print(f"\n=== Training longer run for Best #{i} ===")

        n_states = env.observation_space.n
        n_actions = env.action_space.n
        Q = np.zeros((n_states, n_actions), dtype=float)
        epsilon = params["epsilon"]
        rewards_list = []
        successes_list = []

        for ep in range(LONG_TRAIN_EPISODES):
            obs, _ = env.reset(seed=None)
            state = int(obs)
            total_reward = 0.0
            steps = 0

            action = epsilon_greedy_action(Q, state, n_actions, epsilon)

            while True:
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = int(next_obs)
                done = terminated or truncated

                expected_next = 0.0 if done else _expected_q(Q, next_state, n_actions, epsilon)
                td_target = reward + params["gamma"] * expected_next
                td_error = td_target - Q[state, action]
                Q[state, action] += params["alpha"] * td_error

                state = next_state
                total_reward += reward
                steps += 1

                if done:
                    break

                action = epsilon_greedy_action(Q, state, n_actions, epsilon)

            rewards_list.append(total_reward)
            success = 1 if total_reward > 0 else 0
            successes_list.append(success)

            epsilon = max(params["min_epsilon"], epsilon * params["epsilon_decay"])

            writer.add_scalar("Reward/Episode", total_reward, ep)
            writer.add_scalar("Epsilon", epsilon, ep)
            writer.add_scalar("EpisodeLength", steps, ep)
            writer.add_scalar("Success/Episode", success, ep)

            if ep >= MOVING_AVG_WINDOW:
                moving_avg_reward = np.mean(rewards_list[-MOVING_AVG_WINDOW:])
                writer.add_scalar("Reward/MovingAverage", moving_avg_reward, ep)
                moving_avg_success = np.mean(successes_list[-MOVING_AVG_WINDOW:])
                writer.add_scalar("Success/MovingAverage", moving_avg_success, ep)

            if (ep + 1) % EVAL_INTERVAL == 0:
                eval_result = evaluate_policy(Q, env, n_episodes=10, max_steps=MAX_EPISODE_STEPS)
                writer.add_scalar("Eval/AverageReward", eval_result["average_reward"], ep)
                writer.add_scalar("Eval/AverageSuccess", eval_result["success_rate"], ep)

        np.save(f"report/expected_sarsa/best_{i}_q.npy", Q)
        writer.close()

    print("\nDone. Launch TensorBoard with:\n")
    print("tensorboard --logdir runs/expected_sarsa")


if __name__ == "__main__":
    main()
