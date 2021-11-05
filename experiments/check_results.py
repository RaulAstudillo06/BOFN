import numpy as np
import os
import sys

problem = "covid_calibration"

if problem == "covid_calibration":
    n_evals = 50
    n_trials = 30
    algos = ["KG", "EICF", "EIFN"]

#
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
problem_results_dir = script_dir[:-4] + "experiments_results/" + problem + "/"

print(problem)
for a, algo in enumerate(algos):
    print(algo)

    algo_results_dir = problem_results_dir + algo + "/"
    for trial in range(1, n_trials + 1):
        try:
            trial_n_evals = len(np.loadtxt(algo_results_dir + "best_obs_vals_" + str(trial + 1) + ".txt")) - 1
            if trial_n_evals < n_evals:
                print("Trial {} is not complete yet.".format(trial))
                print("Current number of evaluations is: {}".format(trial_n_evals))
        except:
            print("Trial {} was not found.".format(trial))