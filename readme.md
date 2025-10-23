SARSA:

    execute sarsa_experiment.py: "python -m experiments.sarsa_experiment"

    test frozenlake_custom.py:
        "python -m envs.frozenlake_custom --size 16 --p 0.95 --max_steps 400 --test_episodes 3"
        "python -m envs.frozenlake_custom --size 16 --p 0.95 --save_map report/sarsa/my_map"

    run sarsa.py: "python -m algorithms.sarsa"

    run plot_graphs_sarsa.py: "python -m utils.plot_graphs_sarsa"