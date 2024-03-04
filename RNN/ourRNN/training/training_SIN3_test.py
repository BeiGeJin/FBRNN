# RNN on sine wave, using bp on weight matrix, enforce Dale's law
import numpy as np
import sys
sys.path.append("..")
from rnn_sin3 import RNN
import matplotlib.pyplot as plt
import json
import torch
import torch.nn as nn
import seaborn as sns
from tqdm import tqdm

# Defining constant
ndata = 400
time_constant = 100  # ms
timestep = 10  # ms
time = ndata * timestep  # ms
num_inputs = 1
num_nodes = 32

# Initializing matrix
connectivity_matrix = np.ones((num_nodes, num_nodes))
# weight_matrix = np.ones((num_nodes, num_nodes))
weight_matrix = np.random.normal(size=(num_nodes, num_nodes))
for i in range(num_nodes):
    weight_matrix[i, i] = 0
    connectivity_matrix[i, i] = 0
init_activations = np.zeros((num_nodes, 1))
# init_activations[0] = 1
init_gain = np.ones((num_nodes, 1))
init_shift = np.zeros((num_nodes, 1))
# output_weight_matrix = np.random.normal(size=(1, num_nodes))/num_nodes
output_weight_matrix = np.random.normal(size=(1, num_nodes))

# Create RNN
network = RNN(weight_matrix, connectivity_matrix, init_activations, init_gain, init_shift, output_weight_matrix, 
              time_constant = time_constant, timestep = timestep)

time_points = np.arange(ndata).reshape(-1, 1)
inputs = (1 + np.sin((time_points-30)/60*np.pi))/2
targets = (1 + np.sin((time_points-30+1)/60*np.pi))/2

# training
num_iters = 10000
losses = []

for epoch in tqdm(range(num_iters)):

    loss, _ = network.train_epoch(inputs=inputs, targets=targets, time=time, learning_rate=0.1, mode='weight')
    losses.append(loss)

    # print
    if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{num_iters}, Loss: {loss}\n')
            print(f"{np.sum(network.weight_matrix.detach().numpy())}")