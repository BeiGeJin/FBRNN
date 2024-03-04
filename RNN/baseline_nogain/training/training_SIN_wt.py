# RNN on sine wave, using bp on weight matrix, enforce Dale's law
import matplotlib.pyplot as plt
import numpy as np
from rnn_wt import RNN
import json
from tqdm import tqdm
import os
import torch.nn as nn

num_iters = 5000
num_nodes = 128
# num_iters = int(input("Enter number of training iterations: "))
# num_nodes = int(input("Enter number of nodes: "))

# Defining Inputs and Targets
time_points = np.arange(300).reshape(-1, 1)
# inputs = (1 + np.sin(time_points/60*np.pi))/2
# targets = (1 + np.sin((time_points+1)/60*np.pi))/2
inputs = np.sin(time_points/60*np.pi)
targets = np.sin((time_points+1)/60*np.pi)
# inputs = np.sin(time_points/60*np.pi) / 4 + 0.5
# targets = np.sin((time_points+1)/60*np.pi) / 4 + 0.5
inputs = inputs.reshape(-1, 1)
targets = targets.reshape(-1, 1)
# plt.plot(time_points, inputs, label='input')
# plt.plot(time_points, targets, label='target')
# plt.legend()
# plt.savefig("sin_input.png")

# Defining constant
time_constant = 100  # ms
timestep = 10  # ms
time = 3000  # ms
num_inputs = 1

# Dale's Law
excite_perc = 0.5
excite_num = int(excite_perc*num_nodes)
rng = np.random.default_rng(seed=42)
node_type = rng.permutation([1]*excite_num + [-1]*(num_nodes-excite_num))
weight_type = np.tile(node_type, num_nodes).reshape(num_nodes, -1)

# Initializing matrix
np.random.seed(1)
connectivity_matrix = np.ones((num_nodes, num_nodes))
weight_matrix = np.random.normal(0, 1/np.sqrt(num_nodes), (num_nodes, num_nodes))
init_weight_matrix = weight_matrix.copy()
for i in range(num_nodes):
    weight_matrix[i, i] = 0
    connectivity_matrix[i, i] = 0
input_weight_matrix = np.random.normal(0, 1/np.sqrt(num_inputs), (num_inputs, num_nodes))
init_activations = np.zeros((num_nodes, 1))
init_gains = np.random.normal(0, 1/np.sqrt(num_inputs), (num_nodes, 1))
init_shifts = np.random.normal(0, 1/np.sqrt(num_inputs), (num_nodes, 1))
output_weight_matrix = np.random.normal(0, 1/np.sqrt(num_nodes), (1, num_nodes))

# Enforce Dale's Law
weight_matrix = np.abs(weight_matrix) * weight_type
output_weight_matrix = np.abs(output_weight_matrix) * node_type
# print(weight_matrix)

# Training
losses = []

for epoch in tqdm(range(num_iters), position=0, leave=True):

    # Creating RNN
    network = RNN(weight_matrix, connectivity_matrix, init_activations, init_gains, init_shifts, input_weight_matrix, output_weight_matrix,
                time_constant=time_constant, timestep=timestep)
    
    # train
    loss = network.train_epoch(targets, time, inputs, learning_rate=0.2)

    # update init gains and shifts
    init_gains = network.gain.detach().numpy()
    init_shifts = network.shift.detach().numpy()
    weight_matrix = network.weight_matrix.detach().numpy()

    # weight boundary
    weight_matrix = weight_matrix * connectivity_matrix
    weight_matrix[weight_matrix * weight_type < 0] = 0

    # record
    losses.append(loss)

    # print
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{num_iters}, Loss: {loss}')

# Save
net_weight_history = {}
net_weight_history['trained gain'] = init_gains.tolist()
net_weight_history['trained shift'] = init_shifts.tolist()
net_weight_history['trained weights'] = weight_matrix.tolist()
net_weight_history['connectivity matrix'] = np.asarray(connectivity_matrix).tolist()
net_weight_history['input weights'] = np.asarray(input_weight_matrix).tolist()
net_weight_history['output weights'] = np.asarray(output_weight_matrix).tolist()
net_weight_history['losses'] = np.asarray(losses).tolist()
# net_weight_history['weight_sums'] = np.asarray(weight_history[3]).tolist()
# net_weight_history['gain_changes'] = np.asarray(weight_history[4]).tolist()
net_weight_history['init_weight'] = init_weight_matrix.tolist()

if not os.path.isdir('SIN_wt_' + str(num_nodes) + '_nodes'):
    os.mkdir('SIN_wt_' + str(num_nodes) + '_nodes')
with open('SIN_wt_' + str(num_nodes) + '_nodes/weight_history.json', 'w') as f:
    json.dump(net_weight_history, f)
