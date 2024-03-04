# Linear RNN on MG, bp on weights
# fitting whole MG, not skipping the chaotic period
import sys
sys.path.append("../../model")
import numpy as np
import sys
from rnn_lin import RNN
import json
from tqdm import tqdm
import os
import torch.nn as nn
import matplotlib.pyplot as plt

num_iters = 10000
num_nodes = 32  # stablility of lqr
# num_iters = int(input("Enter number of training iterations: "))
# num_nodes = int(input("Enter number of nodes: "))

# Defining Inputs and Targets
def generate_MackeyGlass(ntimes, tau):
    # initial condition
    # x_values = np.random.uniform(0,1,tau).tolist()
    x_values = [0.1]*tau

    def mackey_glass(x, tau, gamma=0.1, beta0=0.2, n=10):
        dxdt = beta0 * x[-tau] / (1 + x[-tau]**n) - gamma * x[-1]
        return dxdt

    x_t = x_values[-1]
    for t in range(ntimes):
        dxdt = mackey_glass(x_values, tau)
        x_t = x_t + dxdt
        x_values.append(x_t)

    x = x_values[tau:]
    return x

total_time_steps = 600
tau = 20
MG_sequence = np.array(generate_MackeyGlass(total_time_steps+1, tau))
inputs = MG_sequence[0:total_time_steps].reshape(-1,1)
targets = MG_sequence[1:total_time_steps+1].reshape(-1,1)
plt.plot(np.arange(total_time_steps), inputs)
plt.plot(np.arange(total_time_steps), targets)
plt.savefig("MG_input.png")

# Defining constant
time_constant = 100 #ms
timestep = 10 #ms
time = total_time_steps * timestep #ms
num_inputs = 1

# Dale's Law
excite_perc = 0.5
excite_num = int(excite_perc*num_nodes)
rng = np.random.default_rng(seed=42)
node_type = rng.permutation([1]*excite_num + [-1]*(num_nodes-excite_num))

# Initializing matrix
np.random.seed(1)
connectivity_matrix = np.ones((num_nodes, num_nodes))
weight_matrix = np.random.normal(0, 1.2/np.sqrt(num_nodes), (num_nodes, num_nodes))
for i in range(num_nodes):
    weight_matrix[i,i] = 0
    connectivity_matrix[i,i] = 0
input_weight_matrix = np.random.normal(0, 1/np.sqrt(num_inputs), (num_inputs, num_nodes))
init_activations = np.zeros((num_nodes, 1))
init_gains = np.random.normal(0, 1/np.sqrt(num_inputs), (num_nodes, 1))
init_shifts = np.random.normal(0, 1/np.sqrt(num_inputs), (num_nodes, 1))
output_weight_matrix = np.random.normal(0, 1/np.sqrt(num_nodes), (1, num_nodes))

# Enforce Dale's Law
weight_matrix = np.abs(weight_matrix)*(np.tile(node_type, num_nodes).reshape(num_nodes, -1))
output_weight_matrix = np.abs(output_weight_matrix)*node_type

# Creating RNN
network = RNN(weight_matrix, connectivity_matrix, init_activations, init_gains, init_shifts, output_weight_matrix, 
                time_constant = time_constant, timestep = timestep)

#Training Network
net_weight_history = {}
print('Training...', flush = True)
weight_history, losses = network.train(num_iters, targets, time, inputs = inputs,
                                       input_weight_matrix = input_weight_matrix, 
                                       hebbian_learning = False,
                                       learning_rate = .005, fit_start=150)

# net_weight_history['trained gain'] = np.asarray(weight_history[0]).tolist()
# net_weight_history['trained shift'] = np.asarray(weight_history[1]).tolist()
net_weight_history['trained weights'] = np.asarray(weight_history[0]).tolist()
net_weight_history['connectivity matrix'] = np.asarray(connectivity_matrix).tolist()
net_weight_history['input weights'] = np.asarray(input_weight_matrix).tolist()
net_weight_history['output weights'] = np.asarray(output_weight_matrix).tolist()
net_weight_history['losses'] = np.asarray(losses).tolist()
net_weight_history['weight_posneg'] = np.asarray(network.weight_posneg).tolist()

filedir = "../weights/"
filename = "MG_lin_" + str(num_nodes) + "_nodes.json"
filepath = filedir + filename
with open(filepath, 'w') as f:
    json.dump(net_weight_history, f)