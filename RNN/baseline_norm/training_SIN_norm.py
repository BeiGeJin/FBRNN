# training normal RNN which only update weight matrix, with basic sigmoid activation, no gain and shift, no Dale's law
import matplotlib.pyplot as plt
import numpy as np
import sys
from rnn_norm import RNN
import json
from tqdm import tqdm
import os
import torch.nn as nn

num_iters = 3000
num_nodes = 16
# num_iters = int(input("Enter number of training iterations: "))
# num_nodes = int(input("Enter number of nodes: "))

# Defining Inputs and Targets
time_points = np.arange(300).reshape(-1, 1)
# inputs = (1 + np.sin(time_points/10*np.pi))/2
# targets = (1 + np.sin((time_points+1)/10*np.pi))/2
inputs = np.sin(time_points/60*np.pi)
targets = np.sin((time_points+1)/60*np.pi)
inputs = inputs.reshape(-1, 1)
targets = targets.reshape(-1, 1)
# plt.plot(time_points, inputs)
# plt.plot(time_points, targets)
# plt.legend()
# plt.savefig("fig/sin_input.png")

# Defining constant
time_constant = 100  # ms
timestep = 10  # ms
time = 3000  # ms
num_inputs = 1

# Initializing matrix
np.random.seed(1)
connectivity_matrix = np.ones((num_nodes, num_nodes))
weight_matrix = np.random.normal(0, 1/np.sqrt(num_nodes), (num_nodes, num_nodes))
for i in range(num_nodes):
    weight_matrix[i, i] = 0
    connectivity_matrix[i, i] = 0
input_weight_matrix = np.random.normal(0, 1/np.sqrt(num_inputs), (num_inputs, num_nodes))
init_activations = np.zeros((num_nodes, 1))
output_weight_matrix = np.random.normal(0, 1/np.sqrt(num_nodes), (1, num_nodes))
# output_nonlinearity = lambda x: nn.Sigmoid()(x) * 2 - 1

# Creating RNN
network = RNN(weight_matrix, connectivity_matrix, init_activations, output_weight_matrix,
              # output_nonlinearity = lambda x: nn.Sigmoid()(x) * 2 - 1,
              time_constant=time_constant, timestep=timestep)

# Training Network
net_weight_history = {}
print('Training...', flush=True)
weight_history, losses = network.train(num_iters, targets, time, inputs=inputs,
                                       input_weight_matrix=input_weight_matrix, learning_rate=.01, save=5)

net_weight_history['trained weights'] = np.asarray(weight_history[0]).tolist()
net_weight_history['connectivity matrix'] = np.asarray(connectivity_matrix).tolist()
net_weight_history['input weights'] = np.asarray(input_weight_matrix).tolist()
net_weight_history['output weights'] = np.asarray(output_weight_matrix).tolist()

if not os.path.isdir('SIN_norm_' + str(num_nodes) + '_nodes'):
    os.mkdir('SIN_norm_' + str(num_nodes) + '_nodes')
with open('SIN_norm_' + str(num_nodes) + '_nodes/weight_history.json', 'w') as f:
    json.dump(net_weight_history, f)