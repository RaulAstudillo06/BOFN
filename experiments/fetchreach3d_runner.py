import os
import sys
import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.settings import debug
from torch import Tensor

torch.set_default_dtype(torch.float64)
debug._set_state(False)


# Get script directory
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(script_dir[:-12])
print(script_dir[:-12])

from bofn.experiment_manager import experiment_manager
from bofn.utils.dag import DAG
from fetchreach3d import FetchReach


# Function network
n_periods = 3  # Number of periods
problem = "fetchreach3d_" + str(n_periods)  # Problem ID
action_dim = n_periods
state_dim = n_periods
n_nodes = n_periods * state_dim
input_dim = n_periods * action_dim
target_location = torch.tensor([-12.0, 13.0, 0.2])
# target_location = torch.tensor([-12.0, 0.2,   5.0])
# target_location = torch.tensor([16.82, 10.57])
# x_opt = [0.1064, 0.4116, 0.9734, 0.2853, 0.2386, 0.9883]
# x_opt = [0.38356644, 0.42639392, 0.7914563 , 0.38231472, 0.27555513,0.07670888, 0.34121362, 0.52026827, 0.61421557]

fetchreach3d = FetchReach(n_periods=n_periods)


def function_network(X: Tensor) -> Tensor:
    return fetchreach3d.evaluate(X)


# Underlying DAG
dag_as_list = []

for _ in range(state_dim):
    dag_as_list.append([])

for n in range(n_periods - 1):
    for _ in range(state_dim):
        dag_as_list.append([3 * n + j for j in range(state_dim)])

print(dag_as_list)
dag = DAG(dag_as_list)

# Active input indices
active_input_indices = []
for n in range(n_periods):
    for _ in range(state_dim):
        active_input_indices.append([3 * n + j for j in range(action_dim)])
print(active_input_indices)


# Function that maps the network output to the objective value
# network_to_objective_transform = lambda Y: -((Y[:,-2:] - target_location)**2).sum(dim=-1)


def network_to_objective_func(Y, X=None):
    return -((Y[..., -state_dim:] - target_location) ** 2).sum(dim=-1)


network_to_objective_transform = GenericMCObjective(network_to_objective_func)

compute_optimum = False

if compute_optimum:

    def objective_func(x):
        val = -network_to_objective_transform(
            function_network(torch.tensor(x).unsqueeze(dim=0))
        )
        return val.item()

    # print(objective_func(x_opt))
    # print(error)
    bounds = [(0, 1)] * input_dim

    res = optimize.differential_evolution(func=objective_func, bounds=bounds, maxiter=5)
    print(res)
    print(error)


# Run experiment
algo = "KG"

n_bo_iter = 100

if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])

experiment_manager(
    problem=problem,
    algo=algo,
    first_trial=first_trial,
    last_trial=last_trial,
    batch_size=batch_size,
    n_init_evals=2 * (input_dim + 1),
    n_bo_iter=n_bo_iter,
    restart=True,
    function_network=function_network,
    dag=dag,
    active_input_indices=active_input_indices,
    input_dim=input_dim,
    network_to_objective_transform=network_to_objective_transform,
)
