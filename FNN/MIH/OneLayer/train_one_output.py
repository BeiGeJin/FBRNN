import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pickle
import os


class SwinehartAbbottNetworkOneLayer(nn.Module):
    def __init__(self, input_size, init_gain, init_shift, init_weight):
        super(SwinehartAbbottNetworkOneLayer, self).__init__()
        self.input_size = input_size
        self.gain = torch.tensor(init_gain, dtype=torch.float32, requires_grad=True)
        self.shift = torch.tensor(init_shift, dtype=torch.float32, requires_grad=True)
        self.weights = torch.tensor(init_weight, dtype=torch.float32, requires_grad=True)

        self.activation_func = nn.Sigmoid()
        self.gainout = 3
        self.shiftout = 1


    
    def normal_pdf(self, theta):
        return 1.5 * torch.exp(-0.5 * (theta ** 2)) - 0.5

    def gaussian_rf(self, x):
        theta_is = torch.linspace(0, 2 * torch.pi, self.input_size).view(-1,1)
        return self.normal_pdf(x-theta_is) + self.normal_pdf(x-theta_is+2*torch.pi) + self.normal_pdf(x-theta_is-2*torch.pi)

    def forward(self, x):

        x = self.gaussian_rf(x)
        x = self.activation_func(self.gain * (x - self.shift))
        self.input_activation = x.clone()

        x = torch.matmul(self.weights, x)
        x = self.activation_func(self.gainout * (x - self.shiftout))
        self.output_activation = x.clone()

        return x


## RUN
if __name__ == "__main__":
    input_size = 230


    init_gain = 3 * np.ones((input_size, 1))
    init_shift = 1 * np.ones((input_size, 1))
    init_weight = np.ones((1, input_size)) / input_size * 5.5
    torch.manual_seed(42)

    # Data Generation, we will generate data points between 0 and 2*pi
    ndata = 200
    xs = torch.linspace(0, 2 * torch.pi, ndata)
    ys = torch.cos(xs)/4 + 0.5

    # training loop
    num_epochs = 120
  
    hebb_alpha = 5.5
    controller_lr = 0.2
    hebbian_lr = 0.03

    has_controller = True
    has_hebbian = True

    losses = []
    gains = []
    shifts = []
    weights = []
    activations = []
    outputs = []

    # establish model
    model = SwinehartAbbottNetworkOneLayer(input_size, init_gain, init_shift, init_weight)
    loss_func = nn.MSELoss()
    optimizer = optim.SGD([model.gain, model.shift], lr=controller_lr)


    for epoch in range(num_epochs):

        # shuffle data
        perm_idx = torch.randperm(ndata)
        shuffled_xs = xs[perm_idx]
        shuffled_ys = ys[perm_idx]

        epoch_loss = 0


        # forward
        for x, y in zip(shuffled_xs, shuffled_ys):     
 
            # forward   
            actv_opl = model(x)
            output = actv_opl.squeeze()

            # Calculate loss
            loss = 0.5 * loss_func(output, y)
            epoch_loss += loss

            # backprop for gains and shifts
            if has_controller:

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    
            # hebbian learning for weights
            if has_hebbian:

                hebbian_update = model.output_activation * (model.input_activation).T
                model.weights = model.weights + hebbian_lr * hebbian_update
                model.weights = model.weights / torch.sum(model.weights) * hebb_alpha
 
 
        # print losses
        epoch_loss /= ndata
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {epoch_loss}, gain change: {gain_change}, shift change: {shift_change}")


    losses = []
    gains = []
    shifts = []
    weights = []
    activations = []
    outputs = []

        # record
        losses.append(epoch_loss.item())
        gains.append()
        weight_sums.append(np.sum(init_weight))
        gain_changes.append(gain_change)
        shift_changes.append(shift_change)


    filedir = "/Users/marcus/Documents/BouchardLab/FBRNN-MIH_edits/FNN/weights/"
    filename = "weights_abb05_bphebb.pkl"
    filepath = filedir + filename

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
        pickle.dump(losses, f)
        pickle.dump(weight_sums, f)
        pickle.dump(gain_changes, f)
        pickle.dump(shift_changes, f)
        pickle.dump(saved_epoch, f)
        pickle.dump(all_weights, f)