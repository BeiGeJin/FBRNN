# RNN on sine wave, no gain and shift, no sigmoid, only linear, just using bp update weight matrix, to test lqg
import matplotlib.pyplot as plt
import numpy as np
from rnn_lin import RNN
import json
from tqdm import tqdm
import os
import torch.nn as nn

num_iters = 5000
num_nodes = 32  # stablility of lqr
# num_iters = int(input("Enter number of training iterations: "))
# num_nodes = int(input("Enter number of nodes: "))

# Defining Inputs and Targets
time_points = np.arange(300).reshape(-1, 1)
# inputs = (1 + np.sin(time_points/60*np.pi))/2
# targets = (1 + np.sin((time_points+1)/60*np.pi))/2
inputs = np.sin(time_points/60*np.pi)
targets = np.sin((time_points+1)/60*np.pi)
inputs = inputs.reshape(-1, 1)
targets = targets.reshape(-1, 1)
# plt.plot(time_points, inputs)
# plt.plot(time_points, targets)
# plt.savefig("sin_input.png")

# Defining constant
time_constant = 100  # ms
timestep = 10  # ms
time = 3000  # ms
num_inputs = 1

# Dale's Law
excite_perc = 0.50
excite_num = int(excite_perc*num_nodes)
rng = np.random.default_rng(seed=42)
node_type = rng.permutation([1]*excite_num + [-1]*(num_nodes-excite_num))

# Initializing matrix
np.random.seed(1)
connectivity_matrix = np.ones((num_nodes, num_nodes))
weight_matrix = np.random.normal(0, 1/np.sqrt(num_nodes), (num_nodes, num_nodes))
for i in range(num_nodes):
    weight_matrix[i, i] = 0
    connectivity_matrix[i, i] = 0
input_weight_matrix = np.random.normal(0, 1/np.sqrt(num_inputs), (num_inputs, num_nodes))
init_activations = np.zeros((num_nodes, 1))
init_gains = np.random.normal(0, 1/np.sqrt(num_inputs), (num_nodes, 1))
init_shifts = np.random.normal(0, 1/np.sqrt(num_inputs), (num_nodes, 1))
output_weight_matrix = np.random.normal(0, 1/np.sqrt(num_nodes), (1, num_nodes))
# output_nonlinearity = lambda x: nn.Sigmoid()(x) * 2 - 1

# Enforce Dale's Law
weight_matrix = np.abs(weight_matrix)*(np.tile(node_type, num_nodes).reshape(num_nodes, -1))
output_weight_matrix = np.abs(output_weight_matrix)*node_type

# Creating RNN
network = RNN(weight_matrix, connectivity_matrix, init_activations, init_gains, init_shifts, output_weight_matrix,
              # output_nonlinearity = lambda x: nn.Sigmoid()(x) * 2 - 1,
              time_constant=time_constant, timestep=timestep)

# Training Network
net_weight_history = {}
print('Training...', flush=True)
weight_history, losses = network.train(num_iters, targets, time, inputs=inputs,
                                       input_weight_matrix=input_weight_matrix,
                                       hebbian_learning=False, 
                                       learning_rate=.005)

# net_weight_history['trained gain'] = np.asarray(weight_history[0]).tolist()
# net_weight_history['trained shift'] = np.asarray(weight_history[1]).tolist()
# net_weight_history['trained weights'] = np.asarray(weight_history[2]).tolist()
net_weight_history['trained weights'] = np.asarray(weight_history[0]).tolist()
net_weight_history['connectivity matrix'] = np.asarray(connectivity_matrix).tolist()
net_weight_history['input weights'] = np.asarray(input_weight_matrix).tolist()
net_weight_history['output weights'] = np.asarray(output_weight_matrix).tolist()
net_weight_history['losses'] = np.asarray(losses).tolist()
net_weight_history['weight_posneg'] = np.asarray(network.weight_posneg).tolist()

if not os.path.isdir('SIN_lin_' + str(num_nodes) + '_nodes'):
    os.mkdir('SIN_lin_' + str(num_nodes) + '_nodes')
with open('SIN_lin_' + str(num_nodes) + '_nodes/weight_history.json', 'w') as f:
    json.dump(net_weight_history, f)
