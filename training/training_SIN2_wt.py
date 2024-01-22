# RNN on sine wave, using bp on gain, transfer learning to weight
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from rnn_sin2 import RNN
import json
from tqdm import tqdm
import os
import torch.nn as nn
import torch
import pickle

SAVE_CHECKPOINT = False
LOAD_CHECKPOINT = False
checkpoint_epoch = 8000

num_nodes = 32
num_iters = int(input("Enter number of training iterations: "))
# num_nodes = int(input("Enter number of nodes: "))

# Defining Inputs and Targets
ndata = 400
time_points = np.arange(ndata).reshape(-1, 1)
inputs = (1 + np.sin(time_points/60*np.pi))/2
targets = (1 + np.sin((time_points+1)/60*np.pi))/2
inputs = inputs.reshape(-1, 1)
targets = targets.reshape(-1, 1)
plt.plot(time_points, inputs, label='input')
plt.plot(time_points, targets, label='target')
plt.legend()
plt.savefig("sin_input.png")

# Defining constant
time_constant = 100  # ms
timestep = 10  # ms
time = ndata * timestep  # ms
num_inputs = 1

# Dale's Law
excite_perc = 0.5
excite_num = int(excite_perc*num_nodes)
node_type = [1]*excite_num + [-1]*(num_nodes-excite_num)
weight_type = np.tile(node_type, num_nodes).reshape(num_nodes, -1)

# Initializing matrix
np.random.seed(1)
connectivity_matrix = np.ones((num_nodes, num_nodes))
# weight_matrix = np.ones((num_nodes, num_nodes))
weight_matrix = np.random.normal(size=(num_nodes, num_nodes))
for i in range(num_nodes):
    weight_matrix[i, i] = 0
    connectivity_matrix[i, i] = 0
input_weight_matrix = np.random.normal(0, 1/np.sqrt(num_inputs), (num_inputs, num_nodes)) # useless
init_activations = np.zeros((num_nodes, 1))
init_gain = np.ones((num_nodes, 1))
init_shift = np.zeros((num_nodes, 1))
output_weight_matrix = np.ones((1, num_nodes))

# Enforce Dale's Law
weight_matrix = np.abs(weight_matrix) * weight_type
output_weight_matrix = np.abs(output_weight_matrix) * node_type
init_weight_matrix = weight_matrix.copy()

##########
# Training
theo_gain = init_gain.copy()
theo_shift = init_shift.copy()

losses = []
# weights = []
start_pos = 0
has_backprop = True
has_hebbian = False
has_boundary = False
last_epoch_loss = 0

# Load checkpoint
if LOAD_CHECKPOINT:
    with open('checkpoint/checkpoint.pkl', 'rb') as f:
        init_gain = pickle.load(f)
        init_shift = pickle.load(f)
        weight_matrix = pickle.load(f)
    start_pos = checkpoint_epoch

for epoch in tqdm(range(start_pos, num_iters), initial=start_pos, total=num_iters, position=0, leave=True):

    # Creating RNN
    network = RNN(weight_matrix, connectivity_matrix, init_activations, init_gain, init_shift, input_weight_matrix, output_weight_matrix,
                time_constant=time_constant, timestep=timestep, 
                # output_nonlinearity=nn.Sigmoid()
                )
    
    # backprop to update gains and shifts
    if has_backprop:
        loss, activates = network.train_epoch(targets, time, inputs, learning_rate=0.1, mode='weight')
    # weight boundary
    network.weight_matrix = network.weight_matrix * network.connectivity_matrix
    network.weight_matrix[network.weight_matrix * (network.weight_type * 2 - 1) < 0] = 0
    # update init weights
    weight_matrix = network.weight_matrix.detach().numpy()

    # record
    losses.append(loss)
    last_epoch_loss = loss.copy()
    # weights.append(np.sum(weight_matrix))

    # print
    if epoch % 10 == 0:
            excite_weight_sum = np.sum(weight_matrix * (weight_type > 0))
            inhibit_weight_sum = np.sum(weight_matrix * (weight_type < 0))
            print(f'Epoch {epoch+1}/{num_iters}, Loss: {loss}\n\
                  Excite weight sum: {excite_weight_sum}, Inhibit weight sum: {inhibit_weight_sum}')
            # print(network.gain.detach().numpy()[0:10])
    
    # Save checkpoint
    if SAVE_CHECKPOINT and epoch == checkpoint_epoch:
        with open('checkpoint/checkpoint.pkl', 'wb') as f:
            pickle.dump(init_gain, f)
            pickle.dump(init_shift, f)
            pickle.dump(weight_matrix, f) 

# Save
# breakpoint()
net_weight_history = {}
net_weight_history['trained gain'] = init_gain.tolist()
net_weight_history['trained shift'] = init_shift.tolist()
net_weight_history['trained weights'] = weight_matrix.tolist()
net_weight_history['connectivity matrix'] = np.asarray(connectivity_matrix).tolist()
net_weight_history['input weights'] = np.asarray(input_weight_matrix).tolist()
net_weight_history['output weights'] = np.asarray(output_weight_matrix).tolist()
net_weight_history['losses'] = np.asarray(losses).tolist()
# net_weight_history['weights'] = np.asarray(weights).tolist()
net_weight_history['init_weight'] = init_weight_matrix.tolist()

if not os.path.isdir('../weights/SIN2_wt_' + str(num_nodes) + '_nodes'):
    os.mkdir('../weights/SIN2_wt_' + str(num_nodes) + '_nodes')
with open('../weights/SIN2_wt_' + str(num_nodes) + '_nodes/weight_history.json', 'w') as f:
    json.dump(net_weight_history, f)
