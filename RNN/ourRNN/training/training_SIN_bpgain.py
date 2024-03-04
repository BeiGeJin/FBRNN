# RNN on sine wave, just using bp to gain and shift
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from rnn_basic import RNN
import json
from tqdm import tqdm
import os
import torch.nn as nn

num_iters = int(input("Enter number of training iterations: "))  # 20000
num_nodes = int(input("Enter number of nodes: "))  # 128

# Defining Inputs and Targets
ndata = 300
time_points = np.arange(ndata).reshape(-1, 1)
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
time = ndata * timestep  # ms
num_inputs = 1

# Dale's Law
excite_perc = 0.5
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

# record
losses = []

# start
for epoch in tqdm(range(num_iters)):

    # create RNN
    network = RNN(weight_matrix, connectivity_matrix, init_activation, init_gain, init_shift, input_weight_matrix, output_weight_matrix, 
                time_constant=time_constant, timestep=timestep)
    
    # backprop
    loss, activations = network.train_epoch(targets, time, inputs, learning_rate=0.2)

    # update init gain and shift
    init_gain = network.gain.detach().numpy()
    init_shift = network.shift.detach().numpy()

    # record
    losses.append(loss)

    # print
    if epoch % 100 == 0:
        print("Epoch: ", epoch, "Loss: ", loss)

# save
net_weight_history = {}
net_weight_history['trained gain'] = np.asarray(init_gain).tolist()
net_weight_history['trained shift'] = np.asarray(init_shift).tolist()
net_weight_history['trained weights'] = np.asarray(weight_matrix).tolist()
net_weight_history['connectivity matrix'] = np.asarray(connectivity_matrix).tolist()
net_weight_history['input weights'] = np.asarray(input_weight_matrix).tolist()
net_weight_history['output weights'] = np.asarray(output_weight_matrix).tolist()
net_weight_history['losses'] = np.asarray(losses).tolist()
if not os.path.isdir('../weights/SIN_bpgain_' + str(num_nodes) + '_nodes'):
    os.mkdir('../weights/SIN_bpgain_' + str(num_nodes) + '_nodes')
with open('../weights/SIN_bpgain_' + str(num_nodes) + '_nodes/weight_history.json', 'w') as f:
    json.dump(net_weight_history, f)
