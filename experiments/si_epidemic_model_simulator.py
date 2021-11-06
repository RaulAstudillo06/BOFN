import numpy as np

"""
This code runs a SI (no R) epidemiological model with n=2 groups that interact with each other.
Each node is a time period, in which infections occur.


Consider a time period t and let state[i] indicate the fraction of population i infected at the start of this time period.

Each infectious person in group j at the beginning of time period t comes in to contact with beta_ij[t] people from group i.
A fraction (1-state[i]) of these people are susceptible.
Thus, the number of people in group i infected on this time period due to group j is 
(popsize * state[j]) * beta_ij[t] * (1-state[i])

Expressing this as a fraction of group j's population and noting that the population sizes are assumed the same across groups, this is 
(1-state[i]) * beta_ij[t] * state[j] 

Summing across i, we have (1-state[i]) * \sum_j beta_ij[t] * state[j] new infections in group j 

In addition, at the start of the period, a fraction state[i] in population j was infected 

Also, a fraction gamma of the infected people recovered, resulting in a decrease of gamma * state[i] in the fraction infected in population j 

Putting this all together, the new value of state[i] at the end of time period t is

state[i] * (1 - gamma) + (1-state[i]) * \sum_j beta_ij[t] state[j] 
"""


def node(state, beta, gamma):
  # Simulates the next state from the current state and the passed parameters (beta and gamma)
  assert(len(beta) == len(state)) # These should be arrays of the same length
  newstate = np.copy(state)
  for i in range(len(state)):
   newstate[i] = state[i] * (1-gamma) + (1-state[i]) * np.dot(beta[i,:],state)
  return newstate


def simulate(state, beta, gamma):
  # Simulate forward in time, using node and the passed initial state and parameters
  T = len(beta0)
  n = len(state) # number of populations
  assert(np.shape(beta)[0] == T)
  assert(np.shape(beta)[1] == n)
  assert(np.shape(beta)[2] == n)

  history = np.zeros((T,n))
  for t in range(T):
    state = node(state,beta[t],gamma)
    assert(len(state) == n)
    history[t,:] = state # history[t,:] has the state after 

  return history



# Correct values of beta and gamma
# beta is an array of nxn matrices with positive entries
# To make the problem easier, we can restrict to symmetric nxn matrices
# In this matrix, [[A,B], [C,D]],
# A and D are the rate at which people in populations 1 and 2 infect themselves
# Bigger numbers cause infections to grow rapidly
# B is the rate at which population 2 infections population 1
# C is the rate at which population 1 infections population 2

# In this example, transmission is low in the first period for population 1 but
# high for population 2, then high for both in the second, and low for both in
# the third
beta0 = np.asarray([
	[[0.30,0.05], [0.10, 0.7]], 
	[[0.60,0.05], [0.10, 0.8]], 
	[[0.20,0.05], [0.10, 0.2]], 
])
	

# In the example here, population 1's prevalence initially drops, while population 2's rises
# Then, after awhile, population 1's prevalence rises once population 2 has enough infections to start infecting it at a higher rate
"""
beta0 = np.asarray([
	[[0.40,0.05], [0.10, 0.7]], 
	[[0.40,0.05], [0.10, 0.7]], 
	[[0.40,0.05], [0.10, 0.7]], 
	[[0.40,0.05], [0.10, 0.7]], 
	[[0.40,0.05], [0.10, 0.7]], 
	[[0.40,0.05], [0.10, 0.7]], 
	[[0.40,0.05], [0.10, 0.7]], 
	[[0.40,0.05], [0.10, 0.7]], 
	[[0.40,0.05], [0.10, 0.7]], 
	[[0.40,0.05], [0.10, 0.7]], 
	[[0.40,0.05], [0.10, 0.7]], 
	[[0.40,0.05], [0.10, 0.7]], 
	[[0.40,0.05], [0.10, 0.7]], 
	[[0.40,0.05], [0.10, 0.7]], 
	[[0.40,0.05], [0.10, 0.7]], 
])
"""


# gamma is a scalar
gamma0 = 0.5

# Initial state, 1% of each population is infected
state = [0.01, 0.01]

# Correct value for the history
#history0 = simulate(state, beta0, gamma0)
#print(history0)

# To assess performance of a different beta and gamma, measure the sum of squared error between the history produced and history0

# To make the problem higher dimensional, add time periods by adding entries to beta0
# If you'd like to make the problem easier, we can require the values of beta to remain fixed across time periods
