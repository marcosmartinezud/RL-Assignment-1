"""
    TO RUN IT: python -m experiments.sarsa_tensorboard
    Logs rewards, moving average, epsilon, episode length, and success rate to TensorBoard
"""

import os
import numpy as np
import optuna
from torch.utils.tensorboard import SummaryWriter
from algorithms.sarsa import evaluate_policy
from envs.frozenlake_custom import make_frozenlake

# Global environment setup
MAP_SIZE = 16
P_SAFE = 0.95
MAX_EPISODE_STEPS = 400
SEED = 123

# Training parameters
LONG_TRAIN_EPISODES = 20000
EVAL_INTERVAL = 500       # evaluate policy every 500 episodes
MOVING_AVG_WINDOW = 100   # window for moving average reward

def main():
    # Load existing Optuna study
    study = optuna.load_study(
        study_name="sarsa_frozenlake",
        storage="sqlite:///data/sarsa/optuna_sarsa.db"
    )

    # Get top 3 trials
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else 0, reverse=True)[:3]
    print("Top 3 hyperparameter sets:")
    for i, trial in enumerate(top_trials, start=1):
        print(f"Trial {i} params:", trial.params)

    # Prepare logging directory
    base_logdir = "runs/sarsa"
    os.makedirs(base_logdir, exist_ok=True)
    os.makedirs("report/sarsa", exist_ok=True)

    # Create environment
    env, _ = make_frozenlake(
        size=MAP_SIZE,
        p=P_SAFE,
        is_slippery=True,
        max_episode_steps=MAX_EPISODE_STEPS,
        seed=SEED
    )

    # Train each best configuration
    for i, trial in enumerate(top_trials, start=1):
        params = trial.params
        writer = SummaryWriter(log_dir=os.path.join(base_logdir, f"best_{i}"))

        print(f"\n=== Training longer run for Best #{i} ===")

        # Initialize Q-table
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        Q = np.zeros((n_states, n_actions), dtype=float)
        epsilon = params["epsilon"]
        rewards_list = []
        successes_list = []

        for ep in range(LONG_TRAIN_EPISODES):
            obs, _ = env.reset(seed=None)
            state = int(obs)
            # Select initial action for SARSA
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            total_reward = 0.0
            steps = 0

            while True:
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = int(next_obs)
                done = terminated or truncated

                # Choose next action (SARSA policy)
                if np.random.rand() < epsilon:
                    next_action = env.action_space.sample()
                else:
                    next_action = int(np.argmax(Q[next_state]))

                # SARSA update
                td_target = reward + (0.0 if done else params["gamma"] * Q[next_state, next_action])
                td_error = td_target - Q[state, action]
                Q[state, action] += params["alpha"] * td_error

                state = next_state
                action = next_action
                total_reward += reward
                steps += 1

                if done:
                    break

            # Record reward and success
            rewards_list.append(total_reward)
            success = 1 if total_reward > 0 else 0
            successes_list.append(success)

            # Decay epsilon
            epsilon = max(params["min_epsilon"], epsilon * params["epsilon_decay"])

            # Log episode reward
            writer.add_scalar("Reward/Episode", total_reward, ep)
            # Log epsilon value
            writer.add_scalar("Epsilon", epsilon, ep)
            # Log episode length
            writer.add_scalar("EpisodeLength", steps, ep)
            # Log success
            writer.add_scalar("Success/Episode", success, ep)

            # Log moving averages
            if ep >= MOVING_AVG_WINDOW:
                moving_avg_reward = np.mean(rewards_list[-MOVING_AVG_WINDOW:])
                writer.add_scalar("Reward/MovingAverage", moving_avg_reward, ep)
                moving_avg_success = np.mean(successes_list[-MOVING_AVG_WINDOW:])
                writer.add_scalar("Success/MovingAverage", moving_avg_success, ep)

            # Periodic evaluation without exploration
            if (ep + 1) % EVAL_INTERVAL == 0:
                eval_result = evaluate_policy(Q, env, n_episodes=10, max_steps=MAX_EPISODE_STEPS)
                avg_eval_reward = eval_result["average_reward"]
                avg_eval_success = eval_result["success_rate"]
                writer.add_scalar("Eval/AverageReward", avg_eval_reward, ep)
                writer.add_scalar("Eval/AverageSuccess", avg_eval_success, ep)

        # Save final Q-table
        np.save(f"report/sarsa/best_{i}_q.npy", Q)
        writer.close()

    print("\nDone. Launch TensorBoard with:\n")
    print("tensorboard --logdir runs/sarsa")

if __name__ == "__main__":
    main()
