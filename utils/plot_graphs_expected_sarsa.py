"""
    TO RUN IT: python -m utils.plot_graphs_expected_sarsa
"""

import optuna
import optuna.visualization as vis

study = optuna.load_study(
    study_name="expected_sarsa_frozenlake",
    storage="sqlite:///data/expected_sarsa/optuna_expected_sarsa.db",
)

vis.plot_optimization_history(study).show()
vis.plot_param_importances(study).show()
vis.plot_parallel_coordinate(study).show()

