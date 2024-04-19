# Only bp on gains and shifts, no hebbian learning on weight matrix, to test the initial bp
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, input_size1, init_gain, init_gain1, init_shift, init_shift1, init_weight, init_weight1):
        super(SimpleNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.input_size1 = input_size1
        
        self.gain = torch.tensor(init_gain, dtype=torch.float32, requires_grad=True)
        self.gain1 = torch.tensor(init_gain1, dtype=torch.float32, requires_grad=True)
        
        self.shift = torch.tensor(init_shift, dtype=torch.float32, requires_grad=True)
        self.shift1 = torch.tensor(init_shift1, dtype=torch.float32, requires_grad=True)
        
        self.weights = torch.tensor(init_weight, dtype=torch.float32)
        self.weights1 = torch.tensor(init_weight1, dtype=torch.float32, requires_grad=True)
        
        self.activation_func = nn.Sigmoid()

        # just to record
        self.init_gain = torch.tensor(3 * np.ones((self.input_size, 1)), dtype=torch.float32)
        self.init_shift = torch.tensor(1 * np.ones((self.input_size, 1)), dtype=torch.float32)
        self.epoch = 0
    
    def normal_pdf(self, theta):
        # return 1.5 * torch.exp(-0.5 * (theta**2)) - 0.5
        return torch.exp(-0.5 * (theta**2))

    def gaussian_rf(self, x):
        theta_is = torch.linspace(0, 2 * torch.pi, self.input_size).view(-1,1)
        return self.normal_pdf(x - theta_is) + self.normal_pdf(x - theta_is + 2 * torch.pi) + self.normal_pdf(x - theta_is - 2 * torch.pi)

    def forward(self, x):
        x1 = self.gaussian_rf(x)
        self.input_activation = self.activation_func(self.gain * (x1 - self.shift))
        # print(f'final size is {torch.transpose(torch.tensor(self.input_activation),0,1).size()}')
        # self.weights = torch.tensor([[5.5 / (input_size*input_size1)]*input_size1 for _ in range(input_size)])
        # print(len(self.input_activation))
        adjusted_input = torch.transpose(torch.tensor(self.input_activation),0,1)
        x2 = torch.transpose(torch.matmul(adjusted_input, self.weights),0,1)
        """
        x2 is [1 x 50], where 50 is the hidden layer size
        Jerry: I think x2 now is [50 x 1]
        """
        # print(f'completd, size is {x2.size()}')
        self.input_activation1 = self.activation_func(self.gain1 * (x2 - self.shift1))
        
        # print(f'len of each are {self.weights1.size()}, {self.input_activation1.size()}')
        x3 = torch.matmul(self.weights1, self.input_activation1)
        self.output_activation = self.activation_func(3 * (x3 - 1))
        # print(f'final value is {self.output_activation.size()}')
        # print(f'x is {x}')
        # print(f'x1 is {x1}')
        # print(f'input_cat is {self.input_activation}')
        # print(f'x2 is {x2}')
        # print(f'input_cat1 is {self.input_activation1}')
        # print(f'x3 is {x3}')
        # print(f'output_cat is {self.output_activation}')
        return self.output_activation[0][0]

    def train_epoch(self, xs, ys, hebbian_lr = 0.03, hebb_alpha = 5.5, oja_alpha = 1):
        # optimizer = optim.Adam([self.gain, self.shift, self.gain1, self.shift1], lr=0.01)
        optimizer = optim.SGD([self.gain, self.shift, self.gain1, self.shift1], lr=0.2)
        loss_func = nn.MSELoss()
        epoch_loss = 0
        self.epoch += 1

        for x, y in zip(xs, ys):
            optimizer.zero_grad()

            # simulate output
            output = self.forward(x)
            loss = 0.5 * loss_func(output, y)
            loss.backward()
            optimizer.step()

            # print(model.gain.grad)

            # record loss
            epoch_loss += loss

        return epoch_loss.detach().item()


## RUN
if __name__ == "__main__":
    input_size = 100
    input_size1 = 50
    
    init_gain = 3 * np.ones((input_size, 1))
    init_shift = 1 * np.ones((input_size, 1))
    # init_weight = [np.ones((1, input_size1)) * 5.5 / input_size1 for _ in range(input_size)]
    init_weight = np.random.rand(input_size, input_size1) * 5.5 / input_size  # this is to make the activation of each hidden node is different
    
    init_gain1 = 3 * np.ones((input_size1, 1))
    init_shift1 = 1 * np.ones((input_size1, 1))
    # init_weight1 = np.ones((1, input_size1)) * 5.5 / input_size1
    init_weight1 = np.ones((1, input_size1)) / input_size1  # this is to make sure the output sigmoid is not saturated

    theo_gain = 3 * np.ones((input_size, 1))
    theo_shift = 1 * np.ones((input_size, 1))


    # Data Generation, we will generate data points between 0 and 2*pi
    ndata = 200
    xs = torch.linspace(0, 2 * torch.pi, ndata)
    ys = torch.cos(xs)/4 + 0.5

    # Training Loop
    epochs = 200
    losses = []
    gain_changes = []
    shift_changes = []
    gains = []
    shifts = []

    for epoch in tqdm(range(epochs), position=0, leave=True):
        # establish model
        model = SimpleNeuralNetwork(input_size, input_size1, init_gain, init_gain1, init_shift, init_shift1, init_weight, init_weight1)

        # shuffle data
        perm_idx = torch.randperm(ndata)
        shuffled_xs = xs[perm_idx]
        shuffled_ys = ys[perm_idx]
        epoch_loss = model.train_epoch(shuffled_xs, shuffled_ys)
        epoch_loss /= ndata
        losses.append(epoch_loss)

        # update init
        init_gain = model.gain.detach().numpy()
        init_shift = model.shift.detach().numpy()
        init_weight = model.weights.detach().numpy()
        init_gain1 = model.gain1.detach().numpy()
        init_shift1 = model.shift1.detach().numpy()
        init_weight1 = model.weights1.detach().numpy()

        # print out info
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}')
            print(init_gain[0:10])
            # print(init_gain1[0:10])
            # print(init_shift[0:10])
            # print(init_shift1[0:10])
        
        # record
        gain_changes.append(np.linalg.norm(init_gain - theo_gain, 2))
        shift_changes.append(np.linalg.norm(init_shift - theo_shift, 2))
        gains.append(init_gain)
        shifts.append(init_shift)

    # true epoch
    epochs = epoch + 1
    print(f"true epochs: {epochs}")

    filedir = "../weights/"
    filename = "weights_abb05_nohebb_multilayer.pkl"
    filepath = filedir + filename
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
        pickle.dump(losses, f)
        pickle.dump(gain_changes, f)
        pickle.dump(shift_changes, f)
        pickle.dump(gains, f)
        pickle.dump(shifts, f)