# RNN on sine wave, using oja's learning rule to update weight matrix (only excitatory weights)
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from rnn_basic import RNN
import json
from tqdm import tqdm
import os
import torch.nn as nn
import torch

num_iters = int(input("Enter number of training iterations: "))
num_nodes = int(input("Enter number of nodes: "))

# Defining Inputs and Targets
ndata = 300
time_points = np.arange(ndata).reshape(-1, 1)
# inputs = (1 + np.sin(time_points/60*np.pi))/2
# targets = (1 + np.sin((time_points+1)/60*np.pi))/2
inputs = np.sin(time_points/60*np.pi)
targets = np.sin((time_points+1)/60*np.pi)
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
excite_perc = 0.8
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
init_activation = np.zeros((num_nodes, 1))
init_gain = np.random.normal(0, 1/np.sqrt(num_inputs), (num_nodes, 1))
init_shift = np.random.normal(0, 1/np.sqrt(num_inputs), (num_nodes, 1))
output_weight_matrix = np.random.normal(0, 1/np.sqrt(num_nodes), (1, num_nodes))

# Enforce Dale's Law
weight_matrix = np.abs(weight_matrix)*(np.tile(node_type, num_nodes).reshape(num_nodes, -1))
output_weight_matrix = np.abs(output_weight_matrix)*node_type

#######
# Training
#######

# params
theo_gain = init_gain.copy()
theo_shift = init_shift.copy()
oja_alpha = np.sqrt(num_nodes)
hebbian_lr = 0.01

# record
losses = []
gain_changes = []
weight_sums = []

# start
for epoch in tqdm(range(num_iters)):

    # create RNN
    network = RNN(weight_matrix, connectivity_matrix, init_activation, init_gain, init_shift, input_weight_matrix, output_weight_matrix, 
                time_constant=time_constant, timestep=timestep)
    
    # backprop
    loss, activations = network.train_epoch(targets, time, inputs, learning_rate=0.2, optimizer='SGD')

    # update init gain and shift
    init_gain = network.gain.detach().numpy()
    init_shift = network.shift.detach().numpy()
    gain_change = np.linalg.norm(init_gain - theo_gain, 2)

    # oja's learning
    # Calculate Hebbian weight updates
    mean_activates = torch.mean(activations, dim=0).unsqueeze(1)
    hebbian_update = mean_activates * mean_activates.T
    hebbian_update = hebbian_update * network.weight_type * network.connectivity_matrix
    # Regulation term of Oja
    rj_square = (mean_activates**2).expand(-1, network.num_nodes)
    oja_regulation = oja_alpha * rj_square * network.weight_matrix * network.weight_type * network.connectivity_matrix
    # Oja's rule
    network.weight_matrix = network.weight_matrix + hebbian_lr * hebbian_update - hebbian_lr * oja_regulation
    
    # update weight matrix
    weight_matrix = network.weight_matrix.detach().numpy()

    # record
    losses.append(loss)
    gain_changes.append(gain_change)
    weight_sums.append(np.sum(weight_matrix))

    # print
    if epoch % 100 == 0:
        print("Epoch: ", epoch, "Loss: ", loss, "Weight Sum: ", np.sum(weight_matrix))

# save
net_weight_history = {}
net_weight_history['trained gain'] = np.asarray(init_gain).tolist()
net_weight_history['trained shift'] = np.asarray(init_shift).tolist()
net_weight_history['trained weights'] = np.asarray(weight_matrix).tolist()
net_weight_history['connectivity matrix'] = np.asarray(connectivity_matrix).tolist()
net_weight_history['input weights'] = np.asarray(input_weight_matrix).tolist()
net_weight_history['output weights'] = np.asarray(output_weight_matrix).tolist()
net_weight_history['losses'] = np.asarray(losses).tolist()
net_weight_history['weight_sums'] = np.asarray(weight_sums).tolist()
net_weight_history['gain_changes'] = np.asarray(gain_changes).tolist()
if not os.path.isdir('../weights/SIN_bpoja_' + str(num_nodes) + '_nodes'):
    os.mkdir('../weights/SIN_bpoja_' + str(num_nodes) + '_nodes')
with open('../weights/SIN_bpoja_' + str(num_nodes) + '_nodes/weight_history.json', 'w') as f:
    json.dump(net_weight_history, f)
