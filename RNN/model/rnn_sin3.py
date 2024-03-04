import numpy as np
from tqdm import tqdm

import time as pytime
import pdb

import torch
import torch.nn as nn


class RNN:
    '''
    A Class that represents a Recurrent Neural Network with arbitrary struture
    that is trained by implementing gradient descent using the Adam optimizer.
    The output of the network at each timestep is the weighted sum of each of 
    its input nodes with an optional nonlinearity. 

    Attributes
    ----------
    self.weight_matrix : 2D tensor variable
        Represents internal weights of the network.
    self.connectivity_matrix : 2D tensor
        Array of ones and zeros. Only the internal weights 
        corresponding to ones in this matrix can be modified
        during training.
    self.mask : 2D tensor
        Automatically generated "mask" matrix that is used to ensure
        certain weights are constant during training.
    self.output_weight_matrix : 2D tensor
        Weight matrix that represents how to add internal node
        values to create each output. 
    self.activation : 2D tensor
        Column vector of current activations for nodes in the network.
    self.activation_func : pytorch activation function
        Function representing the nonlinearity on each
        internal node. 
    self.output_nonlinearity : pytorch activation function
        Nonlinearity function applied to output. 
    self.time_const : float
        time constant of the decay of signal in eahc node of the
        neural network.
    self.timestep : float 
        timestep of the network.

    '''

    def __init__(self, weight_matrix, connectivity_matrix, init_activations, init_gains, init_shifts,
                 # input_weight_matrix, 
                 output_weight_matrix, output_nonlinearity=lambda x: x,
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
        assert len(weight_matrix.shape) == 2
        assert weight_matrix.shape == connectivity_matrix.shape
        assert weight_matrix.shape[0] == weight_matrix.shape[1]
        assert len(init_activations.shape) == 2
        assert weight_matrix.shape[0] == init_activations.shape[0]
        assert len(output_weight_matrix.shape) == 2
        assert output_weight_matrix.shape[1] == init_activations.shape[0]

        # Parallel pytorch definition - ensure that the gradients are the same
        self.activation = torch.tensor(init_activations, dtype=torch.float32)
        self.init_activation = torch.tensor(init_activations, dtype=torch.float32)
        self.activation_func = activation_func
        self.gain = torch.tensor(init_gains, dtype=torch.float32)
        self.shift = torch.tensor(init_shifts, dtype=torch.float32)
        self.weight_matrix = torch.tensor(weight_matrix, dtype=torch.float32, requires_grad=True)
        self.connectivity_matrix = torch.tensor(connectivity_matrix, dtype=torch.float32)
        # self.mask = torch.eq(self.connectivity_matrix, 0) * self.weight_matrix
        # self.input_weight_matrix = torch.tensor(input_weight_matrix, dtype=torch.float32)
        self.output_weight_matrix = torch.tensor(output_weight_matrix, dtype=torch.float32)
        self.output_nonlinearity = output_nonlinearity

        # constants
        self.time_const = time_constant
        self.timestep = timestep
        self.c = self.timestep/self.time_const
        self.num_nodes = self.weight_matrix.shape[0]
        self.num_outputs = self.output_weight_matrix.shape[0]

        # self.comm = comm
        # if comm is None:
        #     self.rank = 0
        # else:
        #     self.rank = comm.rank

        # Nodes type
        self.weight_type = self.weight_matrix >= 0
        self.node_type = self.weight_type.all(axis=0)

        # just to record
        self.init_gain = init_gains
        self.init_shift = init_shifts

    def normal_pdf(self, theta):
        return torch.exp(-0.5 * (theta**2))

    def input_gaussian(self, x):
        theta_is = torch.linspace(0, 1, self.num_nodes).view(-1,1)
        all_nodes_input = self.normal_pdf(x - theta_is)
        return all_nodes_input
    
    def reset_activations(self):
        self.activation = torch.zeros((self.num_nodes, 1), dtype=torch.float32)

    def forward(self, input):
        # one step forward
        self.layer_input = torch.matmul(self.weight_matrix, self.activation) + self.input_gaussian(input)
        self.activation = (1 - self.c) * self.activation + self.c * self.activation_func(self.gain * (self.layer_input - self.shift))
        output = self.output_nonlinearity(torch.matmul(self.output_weight_matrix, self.activation))[0]
        return output
    
    def simulate(self, time, inputs, disable_progress_bar=False):
        '''
        Simulates timesteps of the network given by time.
        time is equal to num_timesteps_simulated * self.timestep.
        returns all node activations over time and outputs
        over time. 

        Params
        ------
        time : float
            Amount of time to simulate activity. The number of timesteps is given by
            time/self.timestep
        inputs : 2D Tensorflow Matrix Constant
            Matrix where each column is the value of a given input over time. 
        input_weight_matrix : 2D array tensorflow constant
            Weight matrix that represents how to add inputs to each 
            node. Has shape num_inputs x num_nodes. 
        '''
        num_timesteps = int(time//self.timestep)
        activations = []
        outputs = []
        for t in tqdm(range(num_timesteps), position=0, leave=True, disable=disable_progress_bar):
            this_input = inputs[t].item()
            output = self.forward(this_input)
            outputs.append(output)
            activations.append(torch.squeeze(self.activation).clone())
        outputs = torch.stack(outputs, dim=0)
        activations = torch.stack(activations, dim=0)
        return outputs, activations

    def train_epoch(self, targets, time, inputs, learning_rate=0.001, mode='gain'):
            
            inputs = torch.tensor(inputs).float()
            targets = torch.tensor(targets).float()

            # ndata = targets.shape[0]
            # # define target activation
            # target_idxs = (targets * (self.num_nodes-1)).round().int()
            # target_activations = np.zeros((ndata, self.num_nodes))
            # for i, idx in enumerate(target_idxs):
            #     idx = idx[0]
            #     target_activations[i, idx] = 1
            # target_activations = torch.tensor(target_activations).float()

            if mode == 'gain':
                opt = torch.optim.SGD([self.gain, self.shift], lr=learning_rate)
            elif mode == 'weight':
                opt = torch.optim.SGD([self.weight_matrix], lr=learning_rate)
                # opt = torch.optim.Adam([self.weight_matrix], lr=learning_rate)
            self.activation = self.init_activation.clone()
            opt.zero_grad()
            loss_func = nn.MSELoss()
            outputs, activations = self.simulate(time, inputs, disable_progress_bar=True)
            loss_val = loss_func(outputs[100:,:], targets[100:,:])
            # loss_val = loss_func(outputs, targets)
            loss_val.backward()
            opt.step()

            return loss_val.detach().numpy(), activations