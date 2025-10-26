"""
    TO RUN IT: python -m utils.plot_extended_expected_sarsa

    Generates matplotlib figures from the TensorBoard logs stored in
    runs/expected_sarsa/best_{1,2,3}. Two plots are produced:
      - Moving-average reward per run.
      - Episode length per run.
    Images are saved into report/expected_sarsa/.
"""

from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


LOG_DIR = Path("runs/expected_sarsa")
OUTPUT_DIR = Path("report/expected_sarsa")
RUN_NAMES = ["best_1", "best_2", "best_3"]

PLOTS = {
    "Reward/MovingAverage": ("expected_sarsa_moving_average_reward.png", "Moving Average Reward"),
    "EpisodeLength": ("expected_sarsa_episode_length.png", "Episode Length"),
}


def _load_scalar(run: str, tag: str) -> Tuple[List[int], List[float]]:
    run_path = LOG_DIR / run
    if not run_path.exists():
        raise FileNotFoundError(f"No TensorBoard log directory found at {run_path}")

    accumulator = event_accumulator.EventAccumulator(str(run_path))
    accumulator.Reload()

    if tag not in accumulator.Tags().get("scalars", []):
        raise ValueError(f"Tag '{tag}' not found in {run_path}. Available: {accumulator.Tags().get('scalars', [])}")

    events = accumulator.Scalars(tag)
    steps = [event.step for event in events]
    values = [event.value for event in events]
    return steps, values


def _plot_metric(tag: str, filename: str, title: str):
    plt.figure(figsize=(10, 4.5))
    for run in RUN_NAMES:
        steps, values = _load_scalar(run, tag)
        plt.plot(steps, values, label=run.replace("_", " ").title())

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(tag.split("/")[-1])
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {title} plot to {output_path}")


def main():
    for tag, (filename, title) in PLOTS.items():
        _plot_metric(tag, filename, title)


if __name__ == "__main__":
    main()

