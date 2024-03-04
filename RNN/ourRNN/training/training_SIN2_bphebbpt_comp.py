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
checkpoint_epoch = 10000

num_nodes = 32
num_iters = int(input("Enter number of training iterations: "))
# num_nodes = int(input("Enter number of nodes: "))

# Defining Inputs and Targets
period = 120
ndata = period * num_iters
time_points = np.arange(ndata).reshape(-1, 1)
inputs = (1 + np.sin(time_points/60*np.pi))/2
targets = (1 + np.sin((time_points+1)/60*np.pi))/2
inputs = inputs.reshape(-1, 1)
targets = targets.reshape(-1, 1)
# plt.plot(time_points, inputs, label='input')
# plt.plot(time_points, targets, label='target')
# plt.legend()
# plt.savefig("sin_input_all.png")

# Defining constant
time_constant = 100  # ms
timestep = 10  # ms
time = ndata * timestep  # ms
num_inputs = 1
hebb_alpha_ext = 415
hebb_alpha_inh = 375

# Dale's Law
excite_perc = 0.5
excite_num = int(excite_perc*num_nodes)
node_type = [1]*excite_num + [-1]*(num_nodes-excite_num)
weight_type = np.tile(node_type, num_nodes).reshape(num_nodes, -1)

# Initializing matrix
np.random.seed(1)
connectivity_matrix = np.ones((num_nodes, num_nodes))
weight_matrix = np.ones((num_nodes, num_nodes))
for i in range(num_nodes):
    weight_matrix[i, i] = 0
    connectivity_matrix[i, i] = 0
weight_matrix[weight_type > 0] *= hebb_alpha_ext / np.sum(weight_matrix[weight_type > 0])
weight_matrix[weight_type < 0] *= hebb_alpha_inh / np.sum(weight_matrix[weight_type < 0])
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
backprop_lr = 0.1
loss_func = nn.MSELoss()
hebbian_lr = 0.000001
max_hebbian_lr = 0.000001
hebbian_up_rate = max_hebbian_lr / 5000

losses = []
gain_changes = []
shift_changes = []
# weights = []
start_pos = 0
has_backprop = False
has_hebbian = False
has_boundary = False
last_epoch_loss = 0

# Load checkpoint
if LOAD_CHECKPOINT:
    with open('checkpoint/checkpoint_bphebbpt.pkl', 'rb') as f:
        init_gain = pickle.load(f)
        init_shift = pickle.load(f)
        weight_matrix = pickle.load(f)
    start_pos = checkpoint_epoch

# go through all data
for epoch in tqdm(range(start_pos, num_iters), initial=start_pos, total=num_iters, position=0, leave=True):
    
    # skip first epoch
    if epoch > 0 and has_backprop == False:
        has_backprop = True
        print('backprop start!!!')
    
    # start hebbian learning
    if epoch > 1000 and last_epoch_loss < 0.001 and has_hebbian == False:
        has_hebbian = True
        print("hebbian start!!!")
    
    # update hebbian lr if not max
    if has_hebbian:
        if hebbian_lr < max_hebbian_lr:
            hebbian_lr += hebbian_up_rate

    # record
    epoch_losses = []
    epoch_gain_changes = []
    epoch_shift_changes = []

    # reset activation
    # init_activations = np.zeros((num_nodes, 1))

    # go through one period
    for idx in range(period):
        i = epoch * period + idx  # index of data

        # Creating RNN
        network = RNN(weight_matrix, connectivity_matrix, init_activations, init_gain, init_shift, input_weight_matrix, output_weight_matrix,
                    time_constant=time_constant, timestep=timestep)
        
        # forward
        this_input = inputs[i].item()
        this_output = network.forward(this_input).squeeze()
        this_activations = network.activation.clone()

        # get loss
        this_target = torch.tensor(targets[i].item())
        loss_val = loss_func(this_output, this_target)

        # backprop
        if has_backprop:
            opt = torch.optim.SGD([network.gain, network.shift], lr=backprop_lr)
            loss_val.backward()
            opt.step()
            opt.zero_grad()
        init_gain = network.gain.detach().numpy()
        init_shift = network.shift.detach().numpy()
        gain_change = np.linalg.norm(init_gain - theo_gain, 2)
        shift_change = np.linalg.norm(init_shift - theo_shift, 2)

        # update weights by hebbian learning
        if has_hebbian:
            # Calculate Hebbian weight updates
            hebbian_update = torch.matmul(last_activations, this_activations.T)
            hebbian_update = hebbian_update * (network.weight_type * 2 - 1) * network.connectivity_matrix
            # Normalized Hebbian learning
            network.weight_matrix = network.weight_matrix + hebbian_lr * hebbian_update
            network.weight_matrix = network.weight_matrix / torch.sum(torch.abs(network.weight_matrix), dim=0)
            tmp_weights_ext = network.weight_matrix[network.weight_type]
            tmp_weights_inh = network.weight_matrix[~network.weight_type]
            network.weight_matrix[network.weight_type] = tmp_weights_ext * hebb_alpha_ext / 16
            network.weight_matrix[~network.weight_type] = tmp_weights_inh * hebb_alpha_inh / 16
            # weight boundary
            # network.weight_matrix = network.weight_matrix * network.connectivity_matrix
            # network.weight_matrix[network.weight_matrix * (network.weight_type * 2 - 1) < 0] = 0
        # update init weights
        weight_matrix = network.weight_matrix.detach().numpy()

        # update activations
        last_activations = this_activations.clone() # for hebbian
        init_activations = this_activations.detach().numpy() # for next round

        # record
        epoch_losses.append(loss_val.item())
        epoch_gain_changes.append(gain_change)
        epoch_shift_changes.append(shift_change)

    # epoch mean
    mean_epoch_loss = np.mean(epoch_losses)
    mean_epoch_gain_change = np.mean(epoch_gain_changes)
    mean_epoch_shift_change = np.mean(epoch_shift_changes)
    losses.append(mean_epoch_loss)
    gain_changes.append(mean_epoch_gain_change)
    shift_changes.append(mean_epoch_shift_change)
    last_epoch_loss = mean_epoch_loss.copy()
    
    # print
    if epoch % 10 == 0:
            excite_weight_sum = np.sum(weight_matrix * (weight_type > 0))
            inhibit_weight_sum = np.sum(weight_matrix * (weight_type < 0))
            print(f'Epoch {epoch+1}/{num_iters}, Loss: {mean_epoch_loss}, GC:{mean_epoch_gain_change},SC:{mean_epoch_shift_change}, \n\
                  Excite weight sum: {excite_weight_sum}, Inhibit weight sum: {inhibit_weight_sum}')
    
    # Save checkpoint
    if SAVE_CHECKPOINT and epoch + 1 == checkpoint_epoch:
        with open('checkpoint/checkpoint_bphebbpt.pkl', 'wb') as f:
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
net_weight_history['gain_changes'] = np.asarray(gain_changes).tolist()
net_weight_history['shift_changes'] = np.asarray(shift_changes).tolist()
net_weight_history['init_weight'] = init_weight_matrix.tolist()
net_weight_history['init_activations'] = np.asarray(init_activations).tolist()

if not os.path.isdir('../weights/SIN2_bphebbpt_comp_' + str(num_nodes) + '_nodes'):
    os.mkdir('../weights/SIN2_bphebbpt_comp_' + str(num_nodes) + '_nodes')
with open('../weights/SIN2_bphebbpt_comp_' + str(num_nodes) + '_nodes/weight_history.json', 'w') as f:
    json.dump(net_weight_history, f)
