import numpy as np
import os
import sys

problem = "alpine2_6"

if problem == "dropwave":
    n_evals = 100
    n_trials = 30
    algos = ["Random", "EI", "KG", "EIFN"]
elif problem == "alpine2_6":
    n_evals = 100
    n_trials = 30
    algos = ["Random", "EI", "KG", "EIFN"]
elif problem == "robotpush3d_3":
    n_evals = 50
    n_trials = 30
    algos = ["Random", "EI", "KG", "EICF", "EIFN"]
elif problem == "covid_3":
    n_evals = 100
    n_trials = 30
    algos = ["Random", "KG", "EI", "EIFN"]
elif problem == "epidemic_model_calibration":
    n_evals = 50
    n_trials = 30
    algos = ["EIFN"]

#
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
problem_results_dir = script_dir + "/results/" + problem + "/"
print(problem_results_dir)

print(problem)
for a, algo in enumerate(algos):
    print(algo)

    algo_results_dir = problem_results_dir + algo + "/"
    for trial in range(1, n_trials + 1):
        try:
            trial_n_evals = len(np.loadtxt(algo_results_dir + "best_obs_vals_" + str(trial) + ".txt")) - 1
            if trial_n_evals < n_evals:
                print("Trial {} is not complete yet.".format(trial))
                print("Current number of evaluations is: {}".format(trial_n_evals))
        except:
            print("Trial {} was not found.".format(trial))
