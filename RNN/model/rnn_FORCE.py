import numpy as np
from tqdm import tqdm

import time as pytime
import pdb

import torch
import torch.nn as nn


class RNN:
    '''
    RNN for FORCE learning.
    '''

    def __init__(self, weight_matrix, connectivity_matrix, init_state, init_gain, init_shift,
                 output_weight_matrix, feedback_weight_matrix, output_nonlinearity=nn.Sigmoid(),
                 time_constant=1, timestep=0.2, g=1.5, activation_func=nn.Sigmoid(), target=None, gainout=1, shiftout=0):

        # Basic tests to ensure correct input shapes.
        assert len(weight_matrix.shape) == 2
        assert weight_matrix.shape == connectivity_matrix.shape
        assert weight_matrix.shape[0] == weight_matrix.shape[1]
        assert len(init_state.shape) == 2
        assert weight_matrix.shape[0] == init_state.shape[0]
        assert len(output_weight_matrix.shape) == 2
        assert output_weight_matrix.shape[1] == init_state.shape[0]

        # internal weight matrix
        self.weight_matrix = torch.tensor(weight_matrix, dtype=torch.float32)
        self.connectivity_matrix = torch.tensor(connectivity_matrix, dtype=torch.float32)
        self.mask = torch.eq(self.connectivity_matrix, 0) * self.weight_matrix
        # readout weight matrix
        self.output_weight_matrix = torch.tensor(output_weight_matrix, dtype=torch.float32)
        self.output_nonlinearity = output_nonlinearity
        # feedback weight matrix
        self.feedback_weight_matrix = torch.tensor(feedback_weight_matrix, dtype=torch.float32)
        # inside nodes
        self.state = torch.tensor(init_state, dtype=torch.float32)
        self.activation_func = activation_func
        self.gain = torch.tensor(init_gain, dtype=torch.float32, requires_grad=True)
        self.shift = torch.tensor(init_shift, dtype=torch.float32, requires_grad=True)
        self.gainout = gainout
        self.shiftout = shiftout
        # other constant
        self.time_const = time_constant
        self.timestep = timestep
        self.num_nodes = self.weight_matrix.shape[0]
        self.num_outputs = self.output_weight_matrix.shape[0]
        self.g = g
        # Nodes type
        self.weight_type = self.weight_matrix >= 0
        self.node_type = self.weight_type.all(axis=0)
        # whether feedback a target
        self.feed_target = False
        if target is not None:
            self.feed_target = True
            self.target = torch.tensor(target, dtype=torch.float32)
            self.i = 0

    def reset_activations(self):
        self.activation = torch.zeros((self.num_nodes, 1), dtype=torch.float32)

    def forward(self):

        c = self.timestep/self.time_const
        # activation for this state
        self.activation = self.activation_func(self.gain * self.state - self.shift)
        # self.activation = 2 * self.activation_func(self.state) - 1 
        # output for this state
        self.output = self.output_nonlinearity(self.gainout * torch.matmul(self.output_weight_matrix, self.activation) - self.shiftout)
        # feedback for this state
        if not self.feed_target:
            self.feedback = self.output.clone()
        else:
            self.feedback = torch.tensor([[self.target[self.i]]])
            self.i += 1
        # update state
        self.state = (1 - c) * self.state \
            + c * self.g * torch.matmul(self.weight_matrix, self.activation) \
            + c * torch.matmul(self.feedback_weight_matrix, self.feedback)
        
        # breakpoint()
        
        return self.output.detach().item()

    def simulate(self, time):

        num_timesteps = int(time//self.timestep)
        states = []
        activations = []
        outputs = []

        for t in tqdm(range(num_timesteps), position=0, leave=True, disable=True):
            self.forward()
            states.append(self.state.detach().numpy())
            activations.append(self.activation.detach().numpy())
            outputs.append(self.output.detach().item())

        # breakpoint()

        return states, activations, outputs