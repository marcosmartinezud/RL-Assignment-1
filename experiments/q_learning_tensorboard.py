"""
    TO RUN IT: python -m experiments.q_learning_tensorboard
"""

import os
import numpy as np
import optuna
from torch.utils.tensorboard import SummaryWriter
from algorithms.q_learning import train_q_learning
from envs.frozenlake_custom import make_frozenlake

# Global environment setup
MAP_SIZE = 16
P_SAFE = 0.95
MAX_EPISODE_STEPS = 400
SEED = 123

# Training parameters
LONG_TRAIN_EPISODES = 20000
EVAL_INTERVAL = 500  # evaluate policy every 500 episodes
MOVING_AVG_WINDOW = 100  # window for moving average reward

def evaluate_policy(Q, env, n_episodes=10, max_steps=None):
    """Run the policy without exploration to get average reward and success rate"""
    max_steps = max_steps or 1000
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=None)
        state = int(obs)
        total_reward = 0.0
        for _ in range(max_steps):
            action = int(np.argmax(Q[state]))
            obs, reward, terminated, truncated, _ = env.step(action)
            state = int(obs)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return np.mean(rewards)

def main():
    # Load existing Optuna study
    study = optuna.load_study(
        study_name="qlearning_frozenlake",
        storage="sqlite:///data/qlearning/optuna_qlearning.db"
    )

    # Get top 3 trials
    top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:3]
    print("Top 3 hyperparameter sets:")
    for i, trial in enumerate(top_trials, start=1):
        print(f"Trial {i} params:", trial.params)

    # Prepare logging directory
    base_logdir = "runs/qlearning"
    os.makedirs(base_logdir, exist_ok=True)

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
        Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=float)
        epsilon = params["epsilon"]
        rewards_list = []

        for ep in range(LONG_TRAIN_EPISODES):
            obs, _ = env.reset(seed=None)
            state = int(obs)
            total_reward = 0.0
            steps = 0

            while True:
                # Epsilon-greedy action
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = int(np.argmax(Q[state]))

                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = int(next_obs)
                done = terminated or truncated

                # Q-Learning update
                td_target = reward + params["gamma"] * np.max(Q[next_state]) * (0.0 if done else 1.0)
                td_error = td_target - Q[state, action]
                Q[state, action] += params["alpha"] * td_error

                state = next_state
                total_reward += reward
                steps += 1

                if done:
                    break

            rewards_list.append(total_reward)

            # Decay epsilon
            epsilon = max(params["min_epsilon"], epsilon * params["epsilon_decay"])

            # Log episode reward
            writer.add_scalar("Reward/Episode", total_reward, ep)
            # Log epsilon value
            writer.add_scalar("Epsilon", epsilon, ep)
            # Log episode length
            writer.add_scalar("EpisodeLength", steps, ep)

            # Log moving average reward
            if ep >= MOVING_AVG_WINDOW:
                moving_avg = np.mean(rewards_list[-MOVING_AVG_WINDOW:])
                writer.add_scalar("Reward/MovingAverage", moving_avg, ep)

            # Periodic evaluation without exploration
            if (ep + 1) % EVAL_INTERVAL == 0:
                avg_eval_reward = evaluate_policy(Q, env, n_episodes=10, max_steps=MAX_EPISODE_STEPS)
                writer.add_scalar("Eval/AverageReward", avg_eval_reward, ep)

        # Save final Q-table
        np.save(f"report/qlearning/best_{i}_q.npy", Q)
        writer.close()

    print("\nDone. Launch TensorBoard with:\n")
    print("tensorboard --logdir runs/qlearning")

if __name__ == "__main__":
    main()
