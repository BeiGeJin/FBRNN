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
                 output_weight_matrix, output_nonlinearity=lambda x: x,
                 time_constant=1, timestep=0.2, activation_func=nn.Sigmoid(),
                 comm=None):
        '''
        Initializes an instance of the RNN class. 

        Params
        ------
        See Attributes above
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
        # self.weight_matrix = nn.Parameter(torch.from_numpy(weight_matrix).float(), requires_grad=True)
        self.weight_matrix = torch.tensor(weight_matrix, dtype=torch.float32, requires_grad=True)
        self.connectivity_matrix = torch.tensor(connectivity_matrix, dtype=torch.float32)
        self.mask = torch.eq(self.connectivity_matrix, 0) * self.weight_matrix
        self.output_weight_matrix = torch.tensor(output_weight_matrix, dtype=torch.float32)
        self.activation = torch.tensor(init_activations, dtype=torch.float32)
        # self.activation_func = activation_func
        self.activation_func = lambda x: x
        self.output_nonlinearity = output_nonlinearity
        # self.gain = torch.tensor(init_gains, dtype=torch.float32, requires_grad=True)
        # self.shift = torch.tensor(init_shifts, dtype=torch.float32, requires_grad=True)

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
        self.weight_posneg = self.weight_type * 2 - 1
        self.node_type = self.weight_type.all(axis=0)

    def reset_activations(self):
        self.activation = torch.zeros((self.num_nodes, 1), dtype=torch.float32)

    def set_weights(self, internal=None, output=None, connectivity=None):
        '''
        Sets weights of the network
        '''
        # Basic tests to ensure correct input shapes.
        assert len(self.weight_matrix.shape) == 2
        assert self.weight_matrix.shape == self.connectivity_matrix.shape
        assert self.weight_matrix.shape[0] == self.weight_matrix.shape[1]
        assert len(self.output_weight_matrix.shape) == 2

        if internal != None:
            self.weights = internal
        if output != None:
            self.output_weight_matrix = output
        if connectivity != None:
            self.connectivity_matrix = connectivity

        self.mask = torch.eq(self.connectivity_matrix_torch,0) * self.weight_matrix_torch

    @staticmethod
    def convert(time, timestep, funcs=[]):
        '''
        Converts a list of input functions into
        a matrix of different input values over time.
         Makes it easier to create input and
        target matrices for simulation and training of the network. 

        Params
        ------
        time : float
            Amount of time to simulate activity. The number of timesteps is given by
            time/self.timestep
        funcs : list of functions
            Returns value of functions for each time (not per timestep).

        Returns
        -------
        Matrix of input values where each column is an input value over time. 
        '''
        num_timesteps = int(time//timestep)
        result = 0
        if len(funcs) != 0:
            result = [[func(t * timestep) for func in funcs]
                      for t in np.arange(0, num_timesteps, 1)]
            result = torch.tensor(result, dtype=torch.float)
        return result

    def simulate(self, time, inputs=None, input_weight_matrix=None,
                 disable_progress_bar=False):
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
        if input_weight_matrix is not None:
            assert input_weight_matrix.shape[0] == inputs.shape[1]
            assert input_weight_matrix.shape[1] == self.num_nodes
            assert inputs.shape[0] == num_timesteps
            input_weight_matrix = torch.tensor(input_weight_matrix, dtype=torch.float)

        # compiled_activations = torch.zeros((num_timesteps, self.num_nodes))
        # compiled_outputs = torch.zeros((num_timesteps, self.num_outputs))
        compiled_activations = []
        compiled_outputs = []

        c = self.timestep/self.time_const
        
        if input_weight_matrix is not None:
            add_inputs = torch.matmul(inputs, input_weight_matrix)
        else:
            add_inputs = torch.zeros((1, self.num_nodes), dtype=torch.float)

        for t in tqdm(range(num_timesteps), position=0, leave=True, disable=disable_progress_bar):
            # Euler step
            self.activation = (1 - c) * self.activation + \
                c * self.activation_func(torch.matmul(self.weight_posneg * torch.abs(self.weight_matrix), self.activation) +
                                                      torch.unsqueeze(add_inputs[t], 1))

            outputs = self.output_nonlinearity(torch.matmul(self.output_weight_matrix, self.activation))
            # compiled_outputs[t:(t+1)] += outputs
            # compiled_activations[t:(t+1)] += torch.squeeze(self.activation)
            compiled_outputs.append(outputs[0])
            compiled_activations.append(torch.squeeze(self.activation).clone())

        compiled_outputs = torch.stack(compiled_outputs, dim=0)
        compiled_activations = torch.stack(compiled_activations, dim=0)
        # print(compiled_outputs.shape)
        # print(compiled_activations.shape)
        return compiled_outputs, compiled_activations

    def l2_loss_func(self, targets, time, batch_size,
                     inputs, input_weight_matrix,
                     error_mask=None, regularizer=None, fit_start=0):
        '''
        Computes loss function for the given weight matrix.

        Params
        ------
        weight_matrix : tensorflow variable 2D weight matrix
            The weight matrix of the network
        targets : 2D Tensorflow array
            Matrix where each column is the value of a target output over time.
            Should have the same number of columns as the number of outputs. 
        time : float
            Amount of time (not timesteps) the training should simulate output. 
        batch_size : int
            number of trials to run the loss function. Is meant to get an average
            output if the network has stochastic inputs. 
        regularizer : function of the weight matrix (or None)
            regularization term in the loss function. 
        inputs : 2D Tensorflow array
            Matrix where each column is the value of a given input over time. 
        input_weight_matrix : 2D tensorflow constant matrix
            weight matrix for the inputs.
        error_mask : float or 2D tensorflow constant matrix
            Matrix of dimension num_timesteps x num_outputs. Defines
            how to weight network activity for each timestep at each
            output. (Same shape as the inputs and targets matrices).
        '''
        if regularizer == None:
            def regularizer(x): return 0

        num_timesteps = int(time//self.timestep)

        assert targets[0].shape[1] == self.num_outputs

        loss = 0
        for trial in range(batch_size):

            curr_targets = targets[trial]
            curr_inputs = inputs[trial]

            self.reset_activations()

            # simulated, _ = self.simulate(time, curr_inputs,
            #                              input_weight_matrix, True)
            simulated, activates = self.simulate(time, curr_inputs, input_weight_matrix, True)
            self.this_outputs = simulated.clone()  #
            self.this_activates = activates.clone()  #
            l2_ = torch.sum((simulated[fit_start:] - curr_targets[fit_start:])**2)
            loss += 1/(self.num_outputs * num_timesteps) * l2_

        return loss/batch_size + regularizer(self.weight_matrix)

    def train(self, num_iters, targets, time, inputs, input_weight_matrix, batch_size=1,
              learning_rate=0.001, weight_decay=0.002,
              hebbian_learning=True, hebbian_learning_rate=0.01, hebbian_decay=0.999,
              reservoir=False,
              regularizer=None,
              error_mask=None, epochs=10, save=1,
              fit_start=0):
        '''
        Trains the network using l2 loss. See other functions for the definitions of the parameters.
        For this function, instead of having one matrix as inputs/targets/error_mask, the user inputs
        a sequence of matrices. One for each training iteration. This allows for stochasticity in training.
        The parameter save tells us how often to save the weights/loss of the network. A value of 10 would
        result in the weights being saved every ten trials. 
        '''
        weight_history = []
        losses = []

        inputs = torch.tensor(inputs).float()
        targets = torch.tensor(targets).float()
        input_weight_matrix = torch.tensor(input_weight_matrix, dtype=torch.float)

        # if self.rank == 0:
        opt = torch.optim.Adam([self.weight_matrix], lr=learning_rate)
        # opt = torch.optim.Adam([self.gain, self.shift], lr=learning_rate)

        if error_mask == None:
            num_timesteps = int(time//self.timestep)
            error_mask = [torch.ones((num_timesteps, self.num_outputs), dtype=torch.float)] * batch_size

        # Comm size should match the batch size (batch_size)
        # if self.comm is not None:
        #     assert(self.comm.size == batch_size)

        batch_targets = [targets]
        batch_inputs = [inputs]
        batch_error_mask = [error_mask[0]]
        hebbian_lr = hebbian_learning_rate
        oja_alpha = np.sqrt(self.num_nodes)

        for iteration in tqdm(range(num_iters), position=0, leave=True):
            opt.zero_grad()
            loss_val = self.l2_loss_func(batch_targets, time, batch_size,
                                         batch_inputs, input_weight_matrix,
                                         batch_error_mask, regularizer, fit_start=fit_start)
            loss_val.backward()
            opt.step()

            if type(loss_val) == list:
                losses.append(np.mean([l.detach().numpy() for l in loss_val]))
            else:
                losses.append(loss_val.detach().numpy())

            # if iteration % int(num_iters//epochs) == 0:
            #     print("The loss is: " + str(losses[-1]) + " at iteration " + str(iteration), flush = True)
            if (iteration % 100 == 0) or (iteration == num_iters - 1):
                print("The loss is: " + str(losses[-1]) + " at iteration " + str(iteration), flush=True)

            # save weights at last
            if iteration == num_iters - 1:
                # weight_history.append(self.gain.detach().numpy())
                # weight_history.append(self.shift.detach().numpy())
                weight_history.append(self.weight_matrix.detach().numpy())
                if reservoir == True:
                    weight_history.append(self.output_weight_matrix.detach().numpy())

        return weight_history, losses
