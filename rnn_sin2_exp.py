import numpy as np
from tqdm import tqdm

import time as pytime
import pdb

import torch
import torch.nn as nn


class RNN:

    def __init__(self, log_weight_matrix, weight_type, connectivity_matrix, init_activations, init_gains, init_shifts,
                 input_weight_matrix, output_weight_matrix, output_nonlinearity=lambda x: x,
                 time_constant=1, timestep=0.2, activation_func=nn.Sigmoid(),
                 comm=None):
        '''
        Initializes an instance of the RNN class. 

        Params
        ------
        See Attributes above

        In 128 nodes, first 64 ext, last 64 inh. Within 64 each has a gaussian rf for value from 0 to 1, evenly distributed.
        '''
        # Basic tests to ensure correct input shapes.
        assert len(log_weight_matrix.shape) == 2
        assert log_weight_matrix.shape == connectivity_matrix.shape
        assert log_weight_matrix.shape[0] == log_weight_matrix.shape[1]
        assert len(init_activations.shape) == 2
        assert log_weight_matrix.shape[0] == init_activations.shape[0]
        assert len(output_weight_matrix.shape) == 2
        assert output_weight_matrix.shape[1] == init_activations.shape[0]

        # Parallel pytorch definition - ensure that the gradients are the same
        self.log_weight_matrix = torch.tensor(log_weight_matrix, dtype=torch.float32, requires_grad=True)
        self.weight_type = torch.tensor(weight_type, dtype=torch.float32)
        self.connectivity_matrix = torch.tensor(connectivity_matrix, dtype=torch.float32)
        # self.mask = torch.eq(self.connectivity_matrix, 0) * self.weight_matrix
        self.input_weight_matrix = torch.tensor(input_weight_matrix, dtype=torch.float32)
        self.output_weight_matrix = torch.tensor(output_weight_matrix, dtype=torch.float32)
        self.activation = torch.tensor(init_activations, dtype=torch.float32)
        self.activation_func = activation_func
        self.output_nonlinearity = output_nonlinearity
        self.gain = torch.tensor(init_gains, dtype=torch.float32, requires_grad=True)
        self.shift = torch.tensor(init_shifts, dtype=torch.float32, requires_grad=True)

        self.time_const = time_constant
        self.timestep = timestep
        self.num_nodes = self.log_weight_matrix.shape[0]
        self.num_outputs = self.output_weight_matrix.shape[0]

        # Nodes type
        self.node_type = self.weight_type.all(axis=0)

        # just to record
        self.init_gain = init_gains
        self.init_shift = init_shifts

    def normal_pdf(self, theta):
        return torch.exp(-0.5 * (theta**2))

    def input_gaussian(self, x):
        theta_is = torch.linspace(0, 1, int(self.num_nodes/2)).view(-1,1)
        half_nodes_input = self.normal_pdf(x - theta_is)
        all_nodes_input = torch.concat([half_nodes_input, half_nodes_input],dim=0)
        return all_nodes_input
    
    def reset_activations(self):
        self.activation = torch.zeros((self.num_nodes, 1), dtype=torch.float32)

    def forward(self, input):
        # one step forward
        c = self.timestep/self.time_const
        self.layer_input = torch.matmul(torch.exp(self.log_weight_matrix) * self.weight_type, self.activation) + self.input_gaussian(input)
        self.activation = (1 - c) * self.activation + c * self.activation_func(self.gain * (self.layer_input - self.shift))
        output = self.output_nonlinearity(torch.matmul(self.output_weight_matrix, self.activation))
        return output
    
    def simulate(self, time, inputs, disable_progress_bar=False):

        num_timesteps = int(time//self.timestep)
        compiled_activations = []
        compiled_outputs = []
        # c = self.timestep/self.time_const

        for t in tqdm(range(num_timesteps), position=0, leave=True, disable=disable_progress_bar):
            this_input = inputs[t].item()
            output = self.forward(this_input)
            # self.layer_input = torch.matmul(self.weight_matrix, self.activation) + self.input_gaussian(this_input)
            # self.activation = (1 - c) * self.activation + c * self.activation_func(self.gain * (self.layer_input - self.shift))
            # outputs = self.output_nonlinearity(torch.matmul(self.output_weight_matrix, self.activation))
            compiled_outputs.append(output[0])
            compiled_activations.append(torch.squeeze(self.activation).clone())

        compiled_outputs = torch.stack(compiled_outputs, dim=0)
        compiled_activations = torch.stack(compiled_activations, dim=0)

        return compiled_outputs, compiled_activations

    def train_epoch(self, targets, time, inputs, learning_rate=0.001, mode='gain'):
            
        inputs = torch.tensor(inputs).float()
        targets = torch.tensor(targets).float()

        if mode == 'gain':
            opt = torch.optim.SGD([self.gain, self.shift], lr=learning_rate)
        elif mode == 'weight':
            opt = torch.optim.SGD([self.log_weight_matrix], lr=learning_rate)
            # opt = torch.optim.Adam([self.log_weight_matrix], lr=learning_rate)
        self.reset_activations()
        opt.zero_grad()
        loss_func = nn.MSELoss()
        simulated, activates = self.simulate(time, inputs, True)
        loss_val = loss_func(simulated[100:,:], targets[100:,:])
        # loss_val = loss_func(simulated, targets)
        loss_val.backward()
        opt.step()

        return loss_val.detach().numpy(), activates