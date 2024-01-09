# RNN on sine wave, using bp on gain, transfer learning to weight
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from rnn_gn import RNN
import json
from tqdm import tqdm
import os
import torch.nn as nn
import torch
import pickle

SAVE_CHECKPOINT = False
LOAD_CHECKPOINT = True
checkpoint_epoch = 8000

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
excite_perc = 0.5
excite_num = int(excite_perc*num_nodes)
rng = np.random.default_rng(seed=42)
node_type = rng.permutation([1]*excite_num + [-1]*(num_nodes-excite_num))
weight_type = np.tile(node_type, num_nodes).reshape(num_nodes, -1)

# Initializing matrix
np.random.seed(1)
connectivity_matrix = np.ones((num_nodes, num_nodes))
weight_matrix = np.random.normal(0, 1/np.sqrt(num_nodes), (num_nodes, num_nodes))
for i in range(num_nodes):
    weight_matrix[i, i] = 0
    connectivity_matrix[i, i] = 0
input_weight_matrix = np.random.normal(0, 1/np.sqrt(num_inputs), (num_inputs, num_nodes))
init_activations = np.zeros((num_nodes, 1))
init_gain = np.random.normal(0, 1/np.sqrt(num_inputs), (num_nodes, 1))
init_shift = np.random.normal(0, 1/np.sqrt(num_inputs), (num_nodes, 1))
output_weight_matrix = np.random.normal(0, 1/np.sqrt(num_nodes), (1, num_nodes))

# Enforce Dale's Law
weight_matrix = np.abs(weight_matrix) * weight_type
output_weight_matrix = np.abs(output_weight_matrix) * node_type
init_weight_matrix = weight_matrix.copy()
# print(weight_matrix)

# Training
theo_gain = init_gain.copy()
theo_shift = init_shift.copy()
gc_thresh = np.sqrt(num_nodes) * 0.01
sc_thresh = np.sqrt(num_nodes) * 0.01

hebbian_lr = 0
max_hebbian_lr = 0.01
hebbian_up_rate = max_hebbian_lr / 2000
narrow_factor = 0
max_narrow_factor = 0.0001
narrow_up_rate = max_narrow_factor / 2000
oja_alpha_ext = 16.5
oja_alpha_inh = 13.5

losses = []
gain_changes = []
shift_changes = []
# weights = []
start_pos = 0
has_hebbian = 0
has_boundary = 0
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
                time_constant=time_constant, timestep=timestep)
    
    # train
    loss, activates = network.train_epoch(targets, time, inputs, learning_rate=0.2)

    # update init gains and shifts
    init_gain = network.gain.detach().numpy()
    init_shift = network.shift.detach().numpy()
    gain_change = np.linalg.norm(init_gain - theo_gain, 2)
    shift_change = np.linalg.norm(init_shift - theo_shift, 2)

    # weight boundary
    network.weight_matrix = network.weight_matrix * network.connectivity_matrix
    network.weight_matrix[network.weight_matrix * network.weight_type < 0] = 0

    # hebbian learning
    if epoch > 10000 and last_epoch_loss < 0.01 and has_hebbian == 0:
        print("hebbian start!!!")
        has_hebbian = 1
    if has_hebbian == 1 and loss < 0.01:
        # Update hebbian lr if not max
        if hebbian_lr < max_hebbian_lr:
            hebbian_lr += hebbian_up_rate
        # Calculate Hebbian weight updates
        mean_activates = torch.mean(activates, dim=0).unsqueeze(1)
        hebbian_update = mean_activates * mean_activates.T
        # hebbian_update = hebbian_update * network.weight_type * network.connectivity_matrix
        hebbian_update = hebbian_update * (network.weight_type * 2 - 1) * network.connectivity_matrix
        # Regulation term of Oja
        rj_square = (mean_activates**2).expand(-1, network.num_nodes)
        # oja_regulation = oja_alpha * rj_square * network.weight_matrix * network.weight_type * network.connectivity_matrix
        oja_regulation_ext = oja_alpha_ext * rj_square * network.weight_matrix * network.weight_type * network.connectivity_matrix
        oja_regulation_inh = oja_alpha_inh * rj_square * network.weight_matrix * (~network.weight_type) * network.connectivity_matrix
        oja_regulation = oja_regulation_ext + oja_regulation_inh
        # Oja's rule
        network.weight_matrix = network.weight_matrix + hebbian_lr * hebbian_update - hebbian_lr * oja_regulation
    # update init weights
    weight_matrix = network.weight_matrix.detach().numpy()

    # shrink shift and gain to init value
    if epoch > 15000 and last_epoch_loss < 0.005 and has_boundary == 0:
        # create boundaries
        gain_ub = np.maximum(init_gain, theo_gain)
        gain_lb = np.minimum(init_gain, theo_gain)
        shift_ub = np.maximum(init_shift, theo_shift)
        shift_lb = np.minimum(init_shift, theo_shift)
        has_boundary = 1
        print("boundary start!!!")
    if has_boundary == 1 and last_epoch_loss < 0.005:
        # update narrow factor if not max
        if narrow_factor < max_narrow_factor:
            narrow_factor += narrow_up_rate
        # passively narrow the boundaries
        gain_ub = np.maximum(np.minimum(init_gain, gain_ub), theo_gain)
        gain_lb = np.minimum(np.maximum(init_gain, gain_lb), theo_gain)
        shift_ub = np.maximum(np.minimum(init_shift, shift_ub), theo_shift)
        shift_lb = np.minimum(np.maximum(init_shift, shift_lb), theo_shift)
        # actively narrow the boundaries
        if np.linalg.norm(gain_ub - theo_gain, 2) > gc_thresh:
            gain_ub -= narrow_factor * (gain_ub - theo_gain)
        if np.linalg.norm(gain_lb - theo_gain, 2) > gc_thresh:
            gain_lb -= narrow_factor * (gain_lb - theo_gain)
        if np.linalg.norm(shift_ub - theo_shift, 2) > sc_thresh:
            shift_ub -= narrow_factor * (shift_ub - theo_shift)
        if np.linalg.norm(shift_lb - theo_shift, 2) > sc_thresh:
            shift_lb -= narrow_factor * (shift_lb - theo_shift)
    # pull gains and shifts back to into boundaries
    if has_boundary == 1:
        init_gain = np.minimum(init_gain, gain_ub)
        init_gain = np.maximum(init_gain, gain_lb)
        init_shift = np.minimum(init_shift, shift_ub)
        init_shift = np.maximum(init_shift, shift_lb)

    # record
    losses.append(loss)
    last_epoch_loss = loss.copy()
    gain_changes.append(gain_change)
    shift_changes.append(shift_change)
    # weights.append(np.sum(weight_matrix))

    # print
    if epoch % 10 == 0:
            excite_weight_sum = np.sum(weight_matrix * (weight_type > 0))
            inhibit_weight_sum = np.sum(weight_matrix * (weight_type < 0))
            print(f'Epoch {epoch+1}/{num_iters}, Loss: {loss}\n\
                  GC:{gain_change},SC:{shift_change}, WC:{np.sum(weight_matrix)}\n\
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
net_weight_history['gain_changes'] = np.asarray(gain_changes).tolist()
net_weight_history['shift_changes'] = np.asarray(shift_changes).tolist()
net_weight_history['init_weight'] = init_weight_matrix.tolist()

if not os.path.isdir('../weights/sinwave_gn_' + str(num_nodes) + '_nodes'):
    os.mkdir('../weights/sinwave_gn_' + str(num_nodes) + '_nodes')
with open('../weights/sinwave_gn_' + str(num_nodes) + '_nodes/weight_history.json', 'w') as f:
    json.dump(net_weight_history, f)
