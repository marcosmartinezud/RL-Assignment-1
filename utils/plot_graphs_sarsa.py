import optuna
import optuna.visualization as vis

study = optuna.load_study(study_name="sarsa_frozenlake", storage="sqlite:///data/sarsa/optuna_sarsa.db")

vis.plot_optimization_history(study).show()
vis.plot_param_importances(study).show()
vis.plot_parallel_coordinate(study).show()


