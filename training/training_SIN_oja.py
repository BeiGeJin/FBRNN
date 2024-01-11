# RNN on sine wave, using oja's learning rule to update weight matrix (only excitatory weights)
import matplotlib.pyplot as plt
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
time_points = np.arange(300).reshape(-1, 1)
# inputs = (1 + np.sin(time_points/60*np.pi))/2
# targets = (1 + np.sin((time_points+1)/60*np.pi))/2
inputs = np.sin(time_points/60*np.pi)
targets = np.sin((time_points+1)/60*np.pi)
inputs = inputs.reshape(-1, 1)
targets = targets.reshape(-1, 1)
plt.plot(time_points, inputs, label='input')
plt.plot(time_points, targets, label='target')
plt.legend()
plt.savefig("sin_input.png")

# Defining constant
time_constant = 100  # ms
timestep = 10  # ms
time = 3000  # ms
num_inputs = 1

# Dale's Law
excite_perc = 0.8
excite_num = int(excite_perc*num_nodes)
rng = np.random.default_rng(seed=42)
node_type = rng.permutation([1]*excite_num + [-1]*(num_nodes-excite_num))

# Initializing matrix
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
# print(weight_matrix)

# Creating RNN
network = RNN(weight_matrix, connectivity_matrix, init_activations, init_gains, init_shifts, output_weight_matrix,
              # output_nonlinearity = lambda x: nn.Sigmoid()(x) * 2 - 1,
              time_constant=time_constant, timestep=timestep)

# Training Network
net_weight_history = {}
print('Training...', flush=True)
weight_history, losses = network.train(num_iters, targets, time, inputs=inputs,
                                       input_weight_matrix=input_weight_matrix,
                                       hebbian_learning=True, 
                                       learning_rate=.2, hebbian_decay=1)  # 0.2 Adam works well
# breakpoint()

net_weight_history['trained gain'] = np.asarray(weight_history[0]).tolist()
net_weight_history['trained shift'] = np.asarray(weight_history[1]).tolist()
net_weight_history['trained weights'] = np.asarray(weight_history[2]).tolist()
net_weight_history['connectivity matrix'] = np.asarray(connectivity_matrix).tolist()
net_weight_history['input weights'] = np.asarray(input_weight_matrix).tolist()
net_weight_history['output weights'] = np.asarray(output_weight_matrix).tolist()
net_weight_history['losses'] = np.asarray(losses).tolist()
net_weight_history['weight_sums'] = np.asarray(weight_history[3]).tolist()
net_weight_history['gain_changes'] = np.asarray(weight_history[4]).tolist()

if not os.path.isdir('../weights/SIN_oja_' + str(num_nodes) + '_nodes'):
    os.mkdir('../weights/SIN_oja_' + str(num_nodes) + '_nodes')
with open('../weights/SIN_oja_' + str(num_nodes) + '_nodes/weight_history.json', 'w') as f:
    json.dump(net_weight_history, f)
