"""
    TO RUN IT: python -m utils.plot_graphs_qlearning
"""

import optuna
import optuna.visualization as vis

study = optuna.load_study(study_name="qlearning_frozenlake", storage="sqlite:///data/qlearning/optuna_qlearning.db")

vis.plot_optimization_history(study).show()
vis.plot_param_importances(study).show()
vis.plot_parallel_coordinate(study).show()

