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
        assert len(weight_matrix.shape) == 2
        assert weight_matrix.shape == connectivity_matrix.shape
        assert weight_matrix.shape[0] == weight_matrix.shape[1]
        assert len(init_activations.shape) == 2
        assert weight_matrix.shape[0] == init_activations.shape[0]
        assert len(output_weight_matrix.shape) == 2
        assert output_weight_matrix.shape[1] == init_activations.shape[0]

        # Parallel pytorch definition - ensure that the gradients are the same
        self.weight_matrix = torch.tensor(weight_matrix, dtype=torch.float32, requires_grad=True)
        self.connectivity_matrix = torch.tensor(connectivity_matrix, dtype=torch.float32)
        self.mask = torch.eq(self.connectivity_matrix, 0) * self.weight_matrix
        self.input_weight_matrix = torch.tensor(input_weight_matrix, dtype=torch.float32)
        self.output_weight_matrix = torch.tensor(output_weight_matrix, dtype=torch.float32)
        self.activation = torch.tensor(init_activations, dtype=torch.float32)
        self.activation_func = activation_func
        self.output_nonlinearity = output_nonlinearity
        self.gain = torch.tensor(init_gains, dtype=torch.float32, requires_grad=True)
        self.shift = torch.tensor(init_shifts, dtype=torch.float32, requires_grad=True)

        self.time_const = time_constant
        self.timestep = timestep
        self.num_nodes = self.weight_matrix.shape[0]
        self.num_outputs = self.output_weight_matrix.shape[0]

        self.comm = comm
        if comm is None:
            self.rank = 0
        else:
            self.rank = comm.rank

        # Nodes type
        self.weight_type = self.weight_matrix >= 0
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
        self.layer_input = torch.matmul(self.weight_matrix, self.activation) + self.input_gaussian(input)
        self.activation = (1 - c) * self.activation + c * self.activation_func(self.gain * (self.layer_input - self.shift))
        output = self.output_nonlinearity(torch.matmul(self.output_weight_matrix, self.activation))
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
        compiled_activations = []
        compiled_outputs = []
        c = self.timestep/self.time_const

        for t in tqdm(range(num_timesteps), position=0, leave=True, disable=disable_progress_bar):
            this_input = inputs[t].item()
            self.layer_input = torch.matmul(self.weight_matrix, self.activation) + self.input_gaussian(this_input)
            self.activation = (1 - c) * self.activation + c * self.activation_func(self.gain * (self.layer_input - self.shift))
            outputs = self.output_nonlinearity(torch.matmul(self.output_weight_matrix, self.activation))
            compiled_outputs.append(outputs[0])
            compiled_activations.append(torch.squeeze(self.activation).clone())

        compiled_outputs = torch.stack(compiled_outputs, dim=0)
        compiled_activations = torch.stack(compiled_activations, dim=0)

        return compiled_outputs, compiled_activations
    """
    def train(self, num_iters, targets, time, inputs, batch_size=1,
              learning_rate=0.001, weight_decay=0.002,
              hebbian_learning=True, hebbian_learning_rate=0.01, hebbian_decay=0.999):
        '''
        Trains the network using l2 loss. See other functions for the definitions of the parameters.
        For this function, instead of having one matrix as inputs/targets/error_mask, the user inputs
        a sequence of matrices. One for each training iteration. This allows for stochasticity in training.
        The parameter save tells us how often to save the weights/loss of the network. A value of 10 would
        result in the weights being saved every ten trials. 
        '''
        inputs = torch.tensor(inputs).float()
        targets = torch.tensor(targets).float()

        # opt = torch.optim.Adam([self.weight_matrix], lr=learning_rate)
        opt = torch.optim.Adam([self.gain, self.shift], lr=learning_rate)
        # opt = torch.optim.SGD([self.gain, self.shift], lr=learning_rate)
        # hebbian_lr = hebbian_learning_rate
        # oja_alpha = np.sqrt(self.num_nodes)

        weight_history = []
        losses = []
        weight_sums = []
        gain_changes = []

        for iteration in tqdm(range(num_iters), position=0, leave=True):
            # self.reset_activations()
            opt.zero_grad()
            loss_func = nn.MSELoss()
            simulated, _ = self.simulate(time, inputs, True)
            loss_val = loss_func(simulated, targets)
            loss_val.backward()
            opt.step()

            # # Hebbian Learning
            # # average activation rather than single activation!
            # if hebbian_learning == True:
            #     with torch.no_grad():
            #         hebbian_lr *= hebbian_decay
    
            #         # # Calculate Hebbian weight updates
            #         # hebbian_update = self.activation * self.activation.T
            #         # hebbian_update = hebbian_update * self.weight_type * self.connectivity_matrix

            #         # # Regulation term of Oja
            #         # rj_square = (self.activation**2).expand(-1, self.num_nodes)
            #         # oja_regulation = oja_alpha * rj_square * self.weight_matrix * self.weight_type * self.connectivity_matrix
                    
            #         # Calculate Hebbian weight updates
            #         mean_activates = torch.mean(activates, dim=0).unsqueeze(1)
            #         hebbian_update = mean_activates * mean_activates.T
            #         hebbian_update = hebbian_update * self.weight_type * self.connectivity_matrix

            #         # Regulation term of Oja
            #         rj_square = (mean_activates**2).expand(-1, self.num_nodes)
            #         oja_regulation = oja_alpha * rj_square * self.weight_matrix * self.weight_type * self.connectivity_matrix

            #         # # Apply Hebbian updates with a learning rate
            #         # self.weight_matrix = self.weight_matrix + hebbian_lr * hebbian_update
            #         # self.weight_matrix = self.weight_matrix / torch.max(torch.abs(self.weight_matrix)) # normalize with max

            #         # Oja's rule
            #         self.weight_matrix = self.weight_matrix + hebbian_lr * hebbian_update - hebbian_lr * oja_regulation

            # record losses
            if type(loss_val) == list:
                losses.append(np.mean([l.detach().numpy() for l in loss_val]))
            else:
                losses.append(loss_val.detach().numpy())
            
            if (iteration % 100 == 0) or (iteration == num_iters - 1):
                print("The loss is: " + str(losses[-1]) + " at iteration " + str(iteration), flush=True)

            # record weights sum and gain changes
            weight_sums.append(np.sum(self.weight_matrix.detach().numpy()))
            gain_changes.append(np.linalg.norm(self.gain.detach().numpy() - self.init_gain, 2))

            # save weights at last
            if iteration == num_iters - 1:
                weight_history.append(self.gain.detach().numpy())
                weight_history.append(self.shift.detach().numpy())
                weight_history.append(self.weight_matrix.detach().numpy())
                weight_history.append(weight_sums)
                weight_history.append(gain_changes)

        return weight_history, losses
    """

    def train_epoch(self, targets, time, inputs, learning_rate=0.001, mode='gain'):
            
            inputs = torch.tensor(inputs).float()
            targets = torch.tensor(targets).float()

            # opt = torch.optim.Adam([self.weight_matrix], lr=learning_rate)
            if mode == 'gain':
                opt = torch.optim.SGD([self.gain, self.shift], lr=learning_rate)
            elif mode == 'weight':
                opt = torch.optim.SGD([self.weight_matrix], lr=learning_rate)
            self.reset_activations()
            opt.zero_grad()
            loss_func = nn.MSELoss()
            simulated, activates = self.simulate(time, inputs, True)
            loss_val = loss_func(simulated[100:,:], targets[100:,:])
            # loss_val = loss_func(simulated, targets)
            loss_val.backward()
            opt.step()

            return loss_val.detach().numpy(), activates