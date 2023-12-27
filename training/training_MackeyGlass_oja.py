#Training Simple Perceptual Decision Making task
import numpy as np
import sys
sys.path.append("..")
from rnn import RNN
import json
from tqdm import tqdm
import os
import torch.nn as nn

num_iters = int(input("Enter number of training iterations: "))
num_nodes = int(input("Enter number of nodes: "))

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

total_time_steps = 300
tau = 20
MG_sequence = np.array(generate_MackeyGlass(total_time_steps+1, tau))
inputs = MG_sequence[0:300].reshape(-1,1)
targets = MG_sequence[1:301].reshape(-1,1)
import matplotlib.pyplot as plt
plt.plot(np.arange(300), inputs, label='input')
plt.plot(np.arange(300), targets, label='target')
plt.legend()
plt.savefig("MG_input.png")

# Defining constant
time_constant = 100 #ms
timestep = 10 #ms
time = 3000 #ms
num_inputs = 1

# Dale's Law
excite_perc = 0.8
excite_num = int(excite_perc*num_nodes)
rng = np.random.default_rng(seed=42)
node_type = rng.permutation([1]*excite_num + [-1]*(num_nodes-excite_num))

# Initializing matrix
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
                                       hebbian_learning = True,
                                       learning_rate = .005)

net_weight_history['trained gain'] = np.asarray(weight_history[0]).tolist()
net_weight_history['trained shift'] = np.asarray(weight_history[1]).tolist()
net_weight_history['trained weights'] = np.asarray(weight_history[2]).tolist()
net_weight_history['connectivity matrix'] = np.asarray(connectivity_matrix).tolist()
net_weight_history['input weights'] = np.asarray(input_weight_matrix).tolist()
net_weight_history['output weights'] = np.asarray(output_weight_matrix).tolist()
net_weight_history['losses'] = np.asarray(losses).tolist()

if not os.path.isdir('../weights/MacheyGlass_oja_' + str(num_nodes) + '_nodes'):
    os.mkdir('../weights/MacheyGlass_oja_' + str(num_nodes) + '_nodes')
with open('../weights/MacheyGlass_oja_' + str(num_nodes) + '_nodes/weight_history.json', 'w') as f:
    json.dump(net_weight_history, f)