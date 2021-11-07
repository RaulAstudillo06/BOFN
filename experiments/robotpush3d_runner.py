import os
import sys
import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.settings import debug
from scipy import optimize
from torch import Tensor
torch.set_default_dtype(torch.float64)
debug._set_state(True)

# Get script directory
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(script_dir[:-12])

from bofn.experiment_manager import experiment_manager
from bofn.utils.dag import DAG


# Function network
from robotpush3d import RobotPushing3D
n_periods = 3
input_dim = 3 * n_periods
problem = "robotpush3d_" + str(n_periods)

target_location = torch.tensor([2.9287, 1.5983])
x_opt = [0.38356644, 0.42639392, 0.7914563 , 0.38231472, 0.27555513,0.07670888, 0.34121362, 0.52026827, 0.61421557]


robotpush3d = RobotPushing3D(n_periods=n_periods)

def function_network(X: Tensor) -> Tensor:
    return robotpush3d.evaluate(X)


# Underlying DAG
n_nodes = 2 * n_periods
dag_as_list = []
dag_as_list.append([])
dag_as_list.append([])
for n in range(n_periods - 1):
    dag_as_list.append([2 * n, 2 * n + 1])
    dag_as_list.append([2 * n, 2 * n + 1])
print(dag_as_list)
dag= DAG(dag_as_list)

# Active input indices
active_input_indices = []
for n in range(n_periods):
    active_input_indices.append([3 * n + j for j in range(3)])
    active_input_indices.append([3 * n + j for j in range(3)])
print(active_input_indices)
    

# Function that maps the network output to the objective value
#network_to_objective_transform = lambda Y: -((Y[:,-2:] - target_location)**2).sum(dim=-1)

def network_to_objective_func(Y):
    return -((Y[...,-2:] - target_location)**2).sum(dim=-1)
network_to_objective_transform = GenericMCObjective(network_to_objective_func)

compute_optimum = False
if compute_optimum:
    def objective_func(x):
        val = -network_to_objective_transform(function_network(torch.tensor(x).unsqueeze(dim=0)))
        return val.item()

    print(objective_func(x_opt))

    bounds = [(0, 1)] * input_dim

    res = optimize.differential_evolution(func=objective_func, bounds=bounds, maxiter=50)
    print(res)


# Run experiment
algo = "EIFN"

n_bo_iter = 50

if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial =  int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial =  int(sys.argv[1])

experiment_manager(
    problem=problem,
    algo=algo,
    first_trial=first_trial, 
    last_trial=last_trial,
    n_init_evals=2*(input_dim + 1),
    n_bo_iter=n_bo_iter,
    restart=True,
    function_network=function_network,
    dag=dag,
    active_input_indices=active_input_indices,
    input_dim=input_dim,
    network_to_objective_transform=network_to_objective_transform,
)
