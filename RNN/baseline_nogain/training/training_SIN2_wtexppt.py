# RNN on sine wave, using bp on weight matrix, enforce Dale's law (exponential)
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../../model")
from rnn_sin2_exp import RNN
import json
from tqdm import tqdm
import os
import torch.nn as nn
import torch
import pickle

num_nodes = 16
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

# Dale's Law
excite_perc = 0.5
excite_num = int(excite_perc*num_nodes)
node_type = [1]*excite_num + [-1]*(num_nodes-excite_num)
weight_type = np.tile(node_type, num_nodes).reshape(num_nodes, -1)

# Initializing matrix
np.random.seed(1)
connectivity_matrix = np.ones((num_nodes, num_nodes))
log_weight_matrix = np.random.normal(size=(num_nodes, num_nodes))
for i in range(num_nodes):
    # weight_matrix[i, i] = 0
    connectivity_matrix[i, i] = 0
input_weight_matrix = np.random.normal(0, 1/np.sqrt(num_inputs), (num_inputs, num_nodes)) # useless
init_activations = np.zeros((num_nodes, 1))
init_gain = np.ones((num_nodes, 1))
init_shift = np.zeros((num_nodes, 1))
output_weight_matrix = np.ones((1, num_nodes))

# Enforce Dale's Law
output_weight_matrix = np.abs(output_weight_matrix) * node_type
init_weight_matrix = log_weight_matrix.copy()

##########
# Training
theo_gain = init_gain.copy()
theo_shift = init_shift.copy()
backprop_lr = 0.005
loss_func = nn.MSELoss()
losses = []
start_pos = 0
has_backprop = False
last_epoch_loss = 0

# go through all data
for epoch in tqdm(range(start_pos, num_iters), initial=start_pos, total=num_iters, position=0, leave=True):
    
    # skip first epoch
    if epoch > 0 and has_backprop == False:
        has_backprop = True
        print('backprop start!!!')

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
        network = RNN(log_weight_matrix, weight_type, connectivity_matrix, init_activations, init_gain, init_shift, input_weight_matrix, output_weight_matrix,
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
            opt = torch.optim.SGD([network.log_weight_matrix], lr=backprop_lr)
            loss_val.backward()
            opt.step()
            opt.zero_grad()
        
        # update activations
        # last_activations = this_activations.clone() # for hebbian
        init_activations = this_activations.detach().numpy() # for next round

        # update init weights
        log_weight_matrix = network.log_weight_matrix.detach().numpy()

        # record
        epoch_losses.append(loss_val.item())

    # epoch mean
    mean_epoch_loss = np.mean(epoch_losses)
    losses.append(mean_epoch_loss)
    last_epoch_loss = mean_epoch_loss.copy()
    
    # print
    if epoch % 10 == 0:
            excite_weight_sum = np.sum(np.exp(log_weight_matrix) * (weight_type > 0))
            inhibit_weight_sum = np.sum(np.exp(log_weight_matrix) * (weight_type < 0))
            print(f'Epoch {epoch+1}/{num_iters}, Loss: {mean_epoch_loss} \n\
                  Excite weight sum: {excite_weight_sum}, Inhibit weight sum: {inhibit_weight_sum}')

# Save
# breakpoint()
net_weight_history = {}
net_weight_history['trained gain'] = init_gain.tolist()
net_weight_history['trained shift'] = init_shift.tolist()
net_weight_history['trained weights'] = log_weight_matrix.tolist()
net_weight_history['connectivity matrix'] = np.asarray(connectivity_matrix).tolist()
net_weight_history['input weights'] = np.asarray(input_weight_matrix).tolist()
net_weight_history['output weights'] = np.asarray(output_weight_matrix).tolist()
net_weight_history['losses'] = np.asarray(losses).tolist()
net_weight_history['init_weight'] = init_weight_matrix.tolist()
net_weight_history['init_activations'] = np.asarray(init_activations).tolist()

filedir = "../weights/"
filename = "SIN2_wtexppt_" + str(num_nodes) + "_nodes.json"
filepath = filedir + filename
with open(filepath, 'w') as f:
    json.dump(net_weight_history, f)
