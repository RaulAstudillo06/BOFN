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
from covid_simulator import CovidSimulator
n_periods = 3
problem = 'covid_' + str(n_periods)
covid_simulator = CovidSimulator(n_periods=n_periods, seed=1)
input_dim = covid_simulator.input_dim
n_nodes = covid_simulator.n_nodes

def function_network(X: Tensor) -> Tensor:
    return covid_simulator.evaluate(X)

# Underlying DAG
parent_nodes = []
for i in range(3):
    parent_nodes.append([])
for t in range(1, n_periods):
    for i in range(3):
        parent_nodes.append([(t - 1) * 3, (t - 1) * 3 + 1])
dag= DAG(parent_nodes)

# Active input indices
active_input_indices = []

for t in range(n_periods):
    for i in range(3):
        active_input_indices.append([t])

verify_dag_structure = False

if verify_dag_structure:
    print(parent_nodes)
    print(active_input_indices)
    X = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.4], [0.1, 0.3, 0.3]])
    print(function_network(X))

# Function that maps the network output to the objective value

def network_to_objective_transform(Y):
    return -100 * torch.sum(Y[..., [3*t + 2 for t in range(n_periods)]], dim=-1)

network_to_objective_transform = GenericMCObjective(network_to_objective_transform)

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
