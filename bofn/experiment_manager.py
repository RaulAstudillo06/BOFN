from typing import Callable, List, Optional

import os
import sys

from bofn.bofn_trial import bofn_trial
from bofn.utils.dag import DAG


def experiment_manager(
    problem: str,
    algo: str,
    first_trial: int, 
    last_trial: int,
    n_init_evals: int,
    n_bo_iter: int,
    restart: bool,
    function_network: Callable,
    dag: DAG,
    active_input_indices: List[List[Optional[int]]],
    input_dim: int,
    network_to_objective_transform: Callable,
) -> None:
    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    results_folder = script_dir + "/results/" + problem + "/" + algo + "/"

    if not os.path.exists(results_folder) :
        os.makedirs(results_folder)
    if not os.path.exists(results_folder + "runtimes/"):
        os.makedirs(results_folder + "runtimes/")
    if not os.path.exists(results_folder + "X/"):
        os.makedirs(results_folder + "X/")
    if not os.path.exists(results_folder + "network_output_at_X/"):
        os.makedirs(results_folder + "network_output_at_X/")
    if not os.path.exists(results_folder + "objective_at_X/"):
        os.makedirs(results_folder + "objective_at_X/")

    for trial in range(first_trial, last_trial + 1):
        bofn_trial(
            problem=problem,
            function_network=function_network,
            network_to_objective_transform=network_to_objective_transform,
            input_dim=input_dim,
            dag=dag,
            active_input_indices=active_input_indices,
            algo=algo,
            n_init_evals=n_init_evals,
            n_bo_iter=n_bo_iter,
            trial=trial,
            restart=restart,
        )
            