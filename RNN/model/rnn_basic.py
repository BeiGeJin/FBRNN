import numpy as np
from tqdm import tqdm

import time as pytime
import pdb

import torch
import torch.nn as nn


class RNN:
    '''
    Basic RNN, clean code.
    '''

    def __init__(self, weight_matrix, connectivity_matrix, init_activation, init_gain, init_shift,
                 input_weight_matrix, output_weight_matrix, output_nonlinearity=lambda x: x,
                 time_constant=1, timestep=0.2, activation_func=nn.Sigmoid()):

        # Basic tests to ensure correct input shapes.
        assert len(weight_matrix.shape) == 2
        assert weight_matrix.shape == connectivity_matrix.shape
        assert weight_matrix.shape[0] == weight_matrix.shape[1]
        assert len(init_activation.shape) == 2
        assert weight_matrix.shape[0] == init_activation.shape[0]
        assert len(output_weight_matrix.shape) == 2
        assert output_weight_matrix.shape[1] == init_activation.shape[0]

        # Parallel pytorch definition - ensure that the gradients are the same
        self.weight_matrix = torch.tensor(weight_matrix, dtype=torch.float32)
        self.connectivity_matrix = torch.tensor(connectivity_matrix, dtype=torch.float32)
        self.mask = torch.eq(self.connectivity_matrix, 0) * self.weight_matrix
        self.input_weight_matrix = torch.tensor(input_weight_matrix, dtype=torch.float32)
        self.output_weight_matrix = torch.tensor(output_weight_matrix, dtype=torch.float32)
        self.activation = torch.tensor(init_activation, dtype=torch.float32)
        self.activation_func = activation_func
        self.output_nonlinearity = output_nonlinearity
        self.gain = torch.tensor(init_gain, dtype=torch.float32, requires_grad=True)
        self.shift = torch.tensor(init_shift, dtype=torch.float32, requires_grad=True)

        self.time_const = time_constant
        self.timestep = timestep
        self.num_nodes = self.weight_matrix.shape[0]
        self.num_outputs = self.output_weight_matrix.shape[0]

        # Nodes type
        self.weight_type = self.weight_matrix >= 0
        self.node_type = self.weight_type.all(axis=0)

    def reset_activations(self):
        self.activation = torch.zeros((self.num_nodes, 1), dtype=torch.float32)

    def simulate(self, time, inputs):

        num_timesteps = int(time//self.timestep)
        compiled_activations = []
        compiled_outputs = []
        c = self.timestep/self.time_const
        add_inputs = torch.matmul(inputs, self.input_weight_matrix)

        for t in tqdm(range(num_timesteps), position=0, leave=True, disable=True):
            # Euler step
            self.activation = (1 - c) * self.activation + \
                c * self.activation_func(self.gain * (torch.matmul(self.weight_matrix, self.activation) +
                                                      torch.unsqueeze(add_inputs[t], 1) - self.shift))
            output = self.output_nonlinearity(torch.matmul(self.output_weight_matrix, self.activation))
            compiled_outputs.append(output[0])
            compiled_activations.append(torch.squeeze(self.activation).clone())

        compiled_outputs = torch.stack(compiled_outputs, dim=0)
        compiled_activations = torch.stack(compiled_activations, dim=0)

        return compiled_outputs, compiled_activations

    def train_epoch(self, targets, time, inputs, learning_rate=0.2, optimizer='SGD', fit_start=0):
            
        inputs = torch.tensor(inputs).float()
        targets = torch.tensor(targets).float()

        if optimizer == 'SGD':
            opt = torch.optim.SGD([self.gain, self.shift], lr=learning_rate)
        elif optimizer == 'Adam':
            opt = torch.optim.Adam([self.gain, self.shift], lr=learning_rate)
        # opt = torch.optim.SGD([self.weight_matrix], lr=learning_rate)
        self.reset_activations()
        opt.zero_grad()
        loss_func = nn.MSELoss()
        outputs, activations = self.simulate(time, inputs)
        loss_val = loss_func(outputs[fit_start:,:], targets[fit_start:,:])
        loss_val.backward()
        opt.step()

        return loss_val.detach().numpy(), activations