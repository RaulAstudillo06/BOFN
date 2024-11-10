import numpy as np
import os
import sys
import time
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    qExpectedImprovement,
    qKnowledgeGradient,
)
from botorch.acquisition import PosteriorMean as GPPosteriorMean
from botorch.sampling.samplers import SobolQMCNormalSampler
from torch import Tensor
from typing import Callable, List, Optional

from bofn.acquisition_function_optimization.optimize_acqf import (
    optimize_acqf_and_get_suggested_point,
)
from bofn.utils.dag import DAG
from bofn.utils.initial_design import generate_initial_design
from bofn.utils.fit_gp_model import fit_gp_model
from bofn.models.gp_network import GaussianProcessNetwork
from bofn.utils.posterior_mean import PosteriorMean


def bofn_trial(
    problem: str,
    algo: str,
    trial: int,
    batch_size: int,
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

    if restart:
        # Check if training data is already available
        try:
            # Current available evaluations
            X = torch.tensor(np.loadtxt(results_folder + "X/X_" + str(trial) + ".txt"))
            network_output_at_X = torch.tensor(
                np.loadtxt(
                    results_folder
                    + "network_output_at_X/network_output_at_X_"
                    + str(trial)
                    + ".txt"
                )
            )
            objective_at_X = torch.tensor(
                np.loadtxt(
                    results_folder
                    + "objective_at_X/objective_at_X_"
                    + str(trial)
                    + ".txt"
                )
            )

            # Historical best observed objective values and running times
            hist_best_obs_vals = list(
                np.loadtxt(results_folder + "best_obs_vals_" + str(trial) + ".txt")
            )
            runtimes = list(
                np.loadtxt(results_folder + "runtimes/runtimes_" + str(trial) + ".txt")
            )

            # Current best observed objective value
            best_obs_val = torch.tensor(hist_best_obs_vals[-1])

            init_batch_id = len(hist_best_obs_vals)
            print("Restarting experiment from available data.")

        except:

            # Initial evaluations
            X = generate_initial_design(
                num_samples=n_init_evals, input_dim=input_dim, seed=trial
            )
            network_output_at_X = function_network(X)
            objective_at_X = network_to_objective_transform(network_output_at_X)

            # Current best objective value
            best_obs_val = objective_at_X.max().item()

            # Historical best observed objective values and running times
            hist_best_obs_vals = [best_obs_val]
            runtimes = []

            init_batch_id = 1
    else:
        # Initial evaluations
        X = generate_initial_design(
            num_samples=n_init_evals, input_dim=input_dim, seed=trial
        )
        network_output_at_X = function_network(X)
        objective_at_X = network_to_objective_transform(network_output_at_X)

        # Current best objective value
        best_obs_val = objective_at_X.max().item()

        # Historical best observed objective values and running times
        hist_best_obs_vals = [best_obs_val]
        runtimes = []

        init_batch_id = 1

    for iteration in range(init_batch_id, n_bo_iter + 1):
        print("Experiment: " + problem)
        print("Sampling policy: " + algo)
        print("Trial: " + str(trial))
        print("Iteration: " + str(iteration))

        # New suggested point
        t0 = time.time()
        new_x = get_new_suggested_point(
            algo=algo,
            X=X,
            network_output_at_X=network_output_at_X,
            objective_at_X=objective_at_X,
            network_to_objective_transform=network_to_objective_transform,
            dag=dag,
            active_input_indices=active_input_indices,
            batch_size=batch_size,
        )
        t1 = time.time()
        runtimes.append(t1 - t0)

        # Evalaute network at new point
        network_output_at_new_x = function_network(new_x)

        # Evaluate objective at new point
        objective_at_new_x = network_to_objective_transform(network_output_at_new_x)

        # Update training data
        X = torch.cat([X, new_x], 0)
        network_output_at_X = torch.cat(
            [network_output_at_X, network_output_at_new_x], 0
        )
        objective_at_X = torch.cat([objective_at_X, objective_at_new_x], 0)

        # Update historical best observed objective values
        best_obs_val = objective_at_X.max().item()
        hist_best_obs_vals.append(best_obs_val)
        print("Best value found so far: " + str(best_obs_val))

        # Save data
        np.savetxt(results_folder + "X/X_" + str(trial) + ".txt", X.numpy())
        np.savetxt(
            results_folder
            + "network_output_at_X/network_output_at_X_"
            + str(trial)
            + ".txt",
            network_output_at_X.numpy(),
        )
        np.savetxt(
            results_folder + "objective_at_X/objective_at_X_" + str(trial) + ".txt",
            objective_at_X.numpy(),
        )
        np.savetxt(
            results_folder + "best_obs_vals_" + str(trial) + ".txt",
            np.atleast_1d(hist_best_obs_vals),
        )
        np.savetxt(
            results_folder + "runtimes/runtimes_" + str(trial) + ".txt",
            np.atleast_1d(runtimes),
        )


def get_new_suggested_point(
    algo: str,
    X: Tensor,
    network_output_at_X: Tensor,
    objective_at_X: Tensor,
    network_to_objective_transform: Callable,
    dag: DAG,
    active_input_indices: List[int],
    batch_size: int,
) -> Tensor:
    input_dim = X.shape[-1]

    if algo == "Random":
        return torch.rand([1, input_dim])
    elif algo == "EIFN":
        # Model
        model = GaussianProcessNetwork(
            train_X=X,
            train_Y=network_output_at_X,
            dag=dag,
            active_input_indices=active_input_indices,
        )
        # Sampler
        qmc_sampler = SobolQMCNormalSampler(num_samples=128)
        # Acquisition function
        acquisition_function = qExpectedImprovement(
            model=model,
            best_f=objective_at_X.max().item(),
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )
        posterior_mean_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )
    elif algo == "EICF":
        model = fit_gp_model(X=X, Y=network_output_at_X)
        qmc_sampler = SobolQMCNormalSampler(num_samples=128)
        acquisition_function = qExpectedImprovement(
            model=model,
            best_f=objective_at_X.max().item(),
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )
        posterior_mean_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )
    elif algo == "EI":
        model = fit_gp_model(X=X, Y=objective_at_X)
        acquisition_function = ExpectedImprovement(
            model=model, best_f=objective_at_X.max().item()
        )
        posterior_mean_function = GPPosteriorMean(model=model)
    elif algo == "KG":
        model = fit_gp_model(X=X, Y=objective_at_X)
        acquisition_function = qKnowledgeGradient(model=model, num_fantasies=8)
        posterior_mean_function = GPPosteriorMean(model=model)

    new_x = optimize_acqf_and_get_suggested_point(
        acq_func=acquisition_function,
        bounds=torch.tensor(
            [[0.0 for i in range(input_dim)], [1.0 for i in range(input_dim)]]
        ),
        batch_size=batch_size,
        posterior_mean=posterior_mean_function,
    )

    return new_x
