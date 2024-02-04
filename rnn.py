import numpy as np
from tqdm import tqdm

import time as pytime
import pdb

import torch
import torch.nn as nn


class RNN:
    '''
    This RNN is used for initial training: training_bpgain, training_hebb, training_oja.
    The bad thing for this script is complexity.
    The good thing for this scipt is it contains hebbian learning within the class.
    Adam is very fast. But to be consistent with other codes, we use SGD here.
    '''

    def __init__(self, weight_matrix, connectivity_matrix, init_activations, init_gains, init_shifts,
                 output_weight_matrix, output_nonlinearity=lambda x: x,
                 time_constant=1, timestep=0.2, activation_func=nn.Sigmoid(),
                 comm=None):

        # Basic tests to ensure correct input shapes.
        assert len(weight_matrix.shape) == 2
        assert weight_matrix.shape == connectivity_matrix.shape
        assert weight_matrix.shape[0] == weight_matrix.shape[1]
        assert len(init_activations.shape) == 2
        assert weight_matrix.shape[0] == init_activations.shape[0]
        assert len(output_weight_matrix.shape) == 2
        assert output_weight_matrix.shape[1] == init_activations.shape[0]

        # Parallel pytorch definition - ensure that the gradients are the same
        # self.weight_matrix = torch.tensor(weight_matrix, dtype=torch.float32, requires_grad=True)
        self.weight_matrix = torch.tensor(weight_matrix, dtype=torch.float32)
        self.connectivity_matrix = torch.tensor(connectivity_matrix, dtype=torch.float32)
        self.mask = torch.eq(self.connectivity_matrix, 0) * self.weight_matrix
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

    def reset_activations(self):
        self.activation = torch.zeros((self.num_nodes, 1), dtype=torch.float32)

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
                c * self.activation_func(self.gain * (torch.matmul(self.weight_matrix, self.activation) +
                                                      torch.unsqueeze(add_inputs[t], 1) - self.shift))

            outputs = self.output_nonlinearity(torch.matmul(self.output_weight_matrix, self.activation))
            compiled_outputs.append(outputs[0])
            compiled_activations.append(torch.squeeze(self.activation).clone())

        compiled_outputs = torch.stack(compiled_outputs, dim=0)
        compiled_activations = torch.stack(compiled_activations, dim=0)

        return compiled_outputs, compiled_activations

    def l2_loss_func(self, targets, time, batch_size,
                     inputs, input_weight_matrix,
                     error_mask=None, regularizer=None):
        '''
        Computes loss function for the given weight matrix.
        '''
        if regularizer == None:
            def regularizer(x): return 0

        num_timesteps = int(time//self.timestep)

        assert targets[0].shape[1] == self.num_outputs

        loss = 0
        for trial in range(batch_size):

            curr_targets = targets[trial]
            curr_error_mask = error_mask[trial]
            curr_inputs = inputs[trial]

            self.reset_activations()

            simulated, activates = self.simulate(time, curr_inputs, input_weight_matrix, True)
            self.this_outputs = simulated.clone()  #
            self.this_activates = activates.clone()  #
            l2_ = torch.sum((simulated - curr_targets)**2 * curr_error_mask)
            loss += 1/(self.num_outputs * num_timesteps) * l2_

        return loss/batch_size + regularizer(self.weight_matrix)

    def train(self, num_iters, targets, time, inputs, input_weight_matrix, batch_size=1,
              learning_rate=0.001, weight_decay=0.002,
              hebbian_learning=True, hebbian_learning_rate=0.01, hebbian_decay=0.999,
              reservoir=False,
              regularizer=None,
              error_mask=None, epochs=10, save=1):
        '''
        Trains the network using l2 loss. See other functions for the definitions of the parameters.
        For this function, instead of having one matrix as inputs/targets/error_mask, the user inputs
        a sequence of matrices. One for each training iteration. This allows for stochasticity in training.
        The parameter save tells us how often to save the weights/loss of the network. A value of 10 would
        result in the weights being saved every ten trials. 
        '''
        weight_history = []
        losses = []
        weight_sums = []
        gain_changes = []

        inputs = torch.tensor(inputs).float()
        targets = torch.tensor(targets).float()
        input_weight_matrix = torch.tensor(input_weight_matrix, dtype=torch.float)

        # opt = torch.optim.Adam([self.weight_matrix], lr=learning_rate)
        # opt = torch.optim.Adam([self.gain, self.shift], lr=learning_rate)
        opt = torch.optim.SGD([self.gain, self.shift], lr=learning_rate)

        if error_mask == None:
            num_timesteps = int(time//self.timestep)
            error_mask = [torch.ones((num_timesteps, self.num_outputs), dtype=torch.float)] * batch_size

        batch_targets = [targets]
        batch_inputs = [inputs]
        batch_error_mask = [error_mask[0]]
        hebbian_lr = hebbian_learning_rate
        oja_alpha = np.sqrt(self.num_nodes)

        for iteration in tqdm(range(num_iters), position=0, leave=True):
            opt.zero_grad()
            loss_val = self.l2_loss_func(batch_targets, time, batch_size,
                                         batch_inputs, input_weight_matrix,
                                         batch_error_mask, regularizer)
            loss_val.backward(retain_graph=True)
            opt.step()

            # Hebbian Learning
            # average activation rather than single activation!
            if hebbian_learning == True:
                with torch.no_grad():
                    hebbian_lr *= hebbian_decay
    
                    if reservoir == True:
                        # Calculate Hebbian weight updates
                        # outputs = self.output_nonlinearity(torch.matmul(self.output_weight_matrix, self.activation))
                        # hebbian_update = outputs * self.activation.T
                        mean_outputs = torch.mean(self.this_outputs, dim=0).unsqueeze(1)
                        mean_activates = torch.mean(self.this_activates, dim=0).unsqueeze(1)
                        hebbian_update = mean_outputs * mean_activates.T
                        hebbian_update = hebbian_update * self.node_type

                        # Regulation term of Oja
                        # rj_square = (outputs**2).expand(-1, self.num_nodes)
                        rj_square = (mean_outputs**2).expand(-1, self.num_nodes)
                        oja_regulation = oja_alpha * rj_square * self.output_weight_matrix * self.node_type

                        # Oja's rule
                        self.output_weight_matrix = self.output_weight_matrix + hebbian_lr * hebbian_update - hebbian_lr * oja_regulation

                    else:
                        # # Calculate Hebbian weight updates
                        # hebbian_update = self.activation * self.activation.T
                        # hebbian_update = hebbian_update * self.weight_type * self.connectivity_matrix

                        # # Regulation term of Oja
                        # rj_square = (self.activation**2).expand(-1, self.num_nodes)
                        # oja_regulation = oja_alpha * rj_square * self.weight_matrix * self.weight_type * self.connectivity_matrix
                        
                        # Calculate Hebbian weight updates
                        mean_activates = torch.mean(self.this_activates, dim=0).unsqueeze(1)
                        hebbian_update = mean_activates * mean_activates.T
                        hebbian_update = hebbian_update * self.weight_type * self.connectivity_matrix

                        # Regulation term of Oja
                        rj_square = (mean_activates**2).expand(-1, self.num_nodes)
                        oja_regulation = oja_alpha * rj_square * self.weight_matrix * self.weight_type * self.connectivity_matrix

                        # # Apply Hebbian updates with a learning rate
                        # self.weight_matrix = self.weight_matrix + hebbian_lr * hebbian_update
                        # self.weight_matrix = self.weight_matrix / torch.max(torch.abs(self.weight_matrix)) # normalize with max

                        # Oja's rule
                        self.weight_matrix = self.weight_matrix + hebbian_lr * hebbian_update - hebbian_lr * oja_regulation

            # record losses
            if type(loss_val) == list:
                losses.append(np.mean([l.detach().numpy() for l in loss_val]))
            else:
                losses.append(loss_val.detach().numpy())
            
            # print losses
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
                if reservoir == True:
                    weight_history.append(self.output_weight_matrix.detach().numpy())
                weight_history.append(weight_sums)
                weight_history.append(gain_changes)

        return weight_history, losses
