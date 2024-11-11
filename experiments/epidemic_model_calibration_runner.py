import os
import sys
import numpy as np
import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.settings import debug
from torch import Tensor
torch.set_default_dtype(torch.float64)
debug._set_state(True)

# Get script directory
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(script_dir[:-12])

from bofn.experiment_manager import experiment_manager
from bofn.utils.dag import DAG

# Function network
from si_epidemic_model_simulator import simulate

problem = 'epidemic_model_calibration'    
input_dim = 13

state0 = [0.01, 0.01] # initial state, 1% of each population is infected

def function_network(X: Tensor) -> Tensor:
    output = torch.zeros((X.shape[0], 6))
    for i in range(X.shape[0]):
        beta = X[i, :12].view((3, 2, 2))
        gamma = X[i, 12]
        output[i, :] = 100 * torch.tensor(simulate(state0, beta, gamma)).view((6,))
    return output
        
# observed history
beta0 = np.asarray([
	[[0.30,0.05], [0.10, 0.7]], 
	[[0.60,0.05], [0.10, 0.8]], 
	[[0.20,0.05], [0.10, 0.2]], 
])
gamma0 = 0.5
# true underlying parameters
x0 = np.zeros(13)
x0[:12] = beta0.flatten()
x0[12] = gamma0
x0 = torch.tensor(x0).unsqueeze(dim=0)
# observed values
y_true = function_network(x0)

# Underlying DAG
n_nodes = 6
parent_nodes = []
parent_nodes.append([])
parent_nodes.append([])
parent_nodes.append([0, 1])
parent_nodes.append([0, 1])
parent_nodes.append([2, 3])
parent_nodes.append([2, 3])
dag= DAG(parent_nodes=parent_nodes)

# Active input indices
active_input_indices = []
for k in range(3):
    active_input_indices.append([4 * k + j for j in range(4)] + [12])
    active_input_indices.append([4 * k + j for j in range(4)] + [12])
    
# Function that maps the network output to the objective value
network_to_objective_transform = lambda Y: -((Y - y_true)**2).sum(dim=-1)
network_to_objective_transform = GenericMCObjective(network_to_objective_transform)

# Run experiment
algo = "EICF"

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
    batch_size=1,
    n_init_evals=2*(input_dim + 1),
    n_bo_iter=n_bo_iter,
    restart=True,
    function_network=function_network,
    dag=dag,
    active_input_indices=active_input_indices,
    input_dim=input_dim,
    network_to_objective_transform=network_to_objective_transform,
)
