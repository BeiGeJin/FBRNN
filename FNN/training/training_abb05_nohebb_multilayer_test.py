# Only bp on gains and shifts, no hebbian learning on weight matrix, to test the initial bp
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import random

"""
first with 

input:          100 
hidden layer 1: 10
output:         1
"""

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, init_gain, init_shift, init_weight):
        super(SimpleNeuralNetwork, self).__init__()
        self.input_sizes = layer_sizes
        self.gain = torch.tensor(init_gain, dtype=torch.float32, requires_grad=True)
        self.shift = torch.tensor(init_shift, dtype=torch.float32, requires_grad=True)
        self.weights = torch.tensor(init_weight, dtype=torch.float32)
        self.activation_func = nn.Sigmoid()

        # just to record
        self.init_gain = [torch.tensor(3 * np.ones((size, 1)), dtype=torch.float32) for size in self.input_sizes]
        self.init_shift = [torch.tensor(1 * np.ones((size, 1)), dtype=torch.float32) for size in self.input_sizes]
        self.epoch = 0

    def normal_pdf(self, theta):
        # return 1.5 * torch.exp(-0.5 * (theta**2)) - 0.5
        return torch.exp(-0.5 * (theta**2))

    def gaussian_rf(self, x):
        theta_is = torch.linspace(0, 2 * torch.pi, self.input_sizes[0]).view(-1,1)
        return self.normal_pdf(x - theta_is) + self.normal_pdf(x - theta_is + 2 * torch.pi) + self.normal_pdf(x - theta_is - 2 * torch.pi)

    def forward(self, x):
        """
        check activation values here
        """
        index = [self.input_sizes[0], self.input_sizes[1]+self.input_sizes[0]]

        x1 = self.gaussian_rf(x)
        self.l1 = self.activation_func(self.gain[:index[0]] * (x1 - self.shift[:index[0]]))
        
        # print(self.weights[:index[0]])
        
        x2 = torch.matmul(self.weights[:index[0]], self.l1)
        print(x2)
        self.l2 = self.activation_func(self.gain[index[0]:index[1]] * (x2 - self.shift[index[0]:index[1]]))
        
        x3 = torch.matmul(self.weights[index[0]:index[1]], self.l2)
        self.output_activation = self.activation_func(3 * (x3 - 1))
        
        return self.output_activation

    def train_epoch(self, xs, ys, hebbian_lr = 0.03, hebb_alpha = 5.5, oja_alpha = 1):
        """
        weirdly performs well with Adam; due to short computation time?
        """
        # optimizer = optim.Adam([self.gain[0], self.shift[0], self.gain[1], self.shift[1]], lr=0.2)
        optimizer = optim.SGD([self.gain, self.shift], lr=0.2)
        loss_func = nn.MSELoss()
        epoch_loss = 0
        self.epoch += 1
        
        for x, y in zip(xs, ys):
            optimizer.zero_grad()
            # optimizer_1.zero_grad()

            # simulate output
            output = self.forward(x)
            loss = 0.5 * loss_func(output, y)
            loss.backward()
            
            optimizer.step()
            # optimizer_1.step()
            # print(str(self.gain[0][:5])+"\n"+str(self.gain[1][:5])+"\n"+str(self.shift[0][:5])+"\n"+str(self.shift[1][:5])+'\n--------------------')

            # record loss
            epoch_loss += loss

        return epoch_loss.detach().item()


## RUN
if __name__ == "__main__":
    layers = [50, 30]
    tot = sum(layers)
    init_gain = [1,2,3,4,5]*16
    init_shift = [0.2,0.4,0.6,0.8,1]*16
    
    theo_gain = 3
    theo_shift = 1
    init_weight = [5.5/80*n for n in [0.2,0.4,0.6,0.8,1]*16]
    print(len(init_weight))

    # Data Generation, we will generate data points between 0 and 2*pi
    ndata = 200
    xs = torch.linspace(0, 2 * torch.pi, ndata)
    ys = torch.cos(xs)/4 + 0.5

    # Training Loop
    epochs = 200
    losses = []
    
    gain_changes_0 = []
    shift_changes_0 = []
    gain_changes_1 = []
    shift_changes_1 = []
    
    gains_0 = []
    shifts_0 = []
    gains_1 = []
    shifts_1 = []

    for epoch in tqdm(range(epochs), position=0, leave=True):
        # establish model
        model = SimpleNeuralNetwork(layers, init_gain, init_shift, init_weight)

        # shuffle data
        perm_idx = torch.randperm(ndata)
        shuffled_xs = xs[perm_idx]
        shuffled_ys = ys[perm_idx]
        epoch_loss = model.train_epoch(shuffled_xs, shuffled_ys)
        epoch_loss /= ndata
        losses.append(epoch_loss)

        # update init
        init_gain_0 = model.gain[0].detach().numpy()
        init_shift_0 = model.shift[0].detach().numpy()
        # init_weight_0 = model.weights[0].detach().numpy()
        
        init_gain_1 = model.gain[1].detach().numpy()
        init_shift_1 = model.shift[1].detach().numpy()
        # init_weight_1 = model.weights[1].detach().numpy()

        # print out info
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}')
            print(init_gain_0[0:10], init_gain_1[0:10])
        
        # record
        gain_changes_0.append(np.linalg.norm(init_gain_0 - theo_gain, 2))
        shift_changes_0.append(np.linalg.norm(init_shift_0 - theo_shift, 2))
        gains_0.append(init_gain_0)
        shifts_0.append(init_shift_0)
        
        gain_changes_1.append(np.linalg.norm(init_gain_1 - theo_gain, 2))
        shift_changes_1.append(np.linalg.norm(init_shift_1 - theo_shift, 2))
        gains_1.append(init_gain_1)
        shifts_1.append(init_shift_1)

    # true epoch
    epochs = epoch + 1
    print(f"true epochs: {epochs}")

    filedir = "../weights/"
    filename = "weights_abb05_nohebb_multilayer_test.pkl"
    filepath = filedir + filename
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
        pickle.dump(losses, f)
        
        pickle.dump(gain_changes_0, f)
        pickle.dump(shift_changes_0, f)
        pickle.dump(gains_0, f)
        pickle.dump(shifts_0, f)
        
        pickle.dump(gain_changes_1, f)
        pickle.dump(shift_changes_1, f)
        pickle.dump(gains_1, f)
        pickle.dump(shifts_1, f)