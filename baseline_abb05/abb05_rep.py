import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, init_gain, init_shift, init_weight):
        super(SimpleNeuralNetwork, self).__init__()
        self.input_size = input_size
        # init_gain = 3 * np.ones((self.input_size, 1))
        # init_shift = 1 * np.ones((self.input_size, 1))
        # init_weight = np.ones((1,self.input_size))
        self.gain = torch.tensor(init_gain, dtype=torch.float32, requires_grad=True)
        self.shift = torch.tensor(init_shift, dtype=torch.float32, requires_grad=True)
        self.weights = torch.tensor(init_weight, dtype=torch.float32)
        self.activation_func = nn.Sigmoid()

        # just to record
        self.init_gain = torch.tensor(3 * np.ones((self.input_size, 1)), dtype=torch.float32)
        self.init_shift = torch.tensor(1 * np.ones((self.input_size, 1)), dtype=torch.float32)
    
    def normal_pdf(self, theta):
        # return 1.5 * torch.exp(-0.5 * theta ** 2) - 0.5
        return torch.exp(-0.5 * (theta**2))

    def gaussian_rf(self, x):
        theta_is = torch.linspace(0, 2 * torch.pi, self.input_size).view(-1,1)
        return self.normal_pdf(x-theta_is) + self.normal_pdf(x-theta_is+2*torch.pi) + self.normal_pdf(x-theta_is-2*torch.pi)

    def forward(self, x):
        x = self.gaussian_rf(x)
        x = self.activation_func(self.gain * (x - self.shift))
        self.input_activation = x.clone()
        x = torch.matmul(self.weights, x)
        x = self.activation_func(3 * (x - 1))
        self.output_activation = x.clone()
        return x

    def train_point(self, x, y):
        # optimizer = optim.Adam([self.gain, self.shift], lr=0.05)
        optimizer = optim.SGD([self.gain, self.shift], lr=0.2)
        loss_func = nn.MSELoss()
        optimizer.zero_grad()

        # simulate output
        output = self.forward(x).squeeze()
        loss = 0.5 * loss_func(output, y)
        # loss = loss + 0.0000001 * epoch * (torch.linalg.vector_norm(self.gain - self.init_gain) + torch.linalg.vector_norm(self.shift - self.init_shift))
        
        # gain modulation
        loss.backward()
        optimizer.step()
        
        return loss.detach().item()


## RUN
if __name__ == "__main__":
    input_size = 230
    theo_gain = 3 * np.ones((input_size, 1))
    theo_shift = 1 * np.ones((input_size, 1))
    init_gain = 3 * np.ones((input_size, 1))
    init_shift = 1 * np.ones((input_size, 1))
    init_weight = np.ones((1, input_size)) / input_size * 5.5

    # Data Generation, we will generate data points between 0 and 2*pi
    ndata = 200
    xs = torch.linspace(0, 2 * torch.pi, ndata)
    ys = torch.cos(xs)/4 + 0.5

    # hebbian_lr = 0.03
    # hebb_alpha = 5.5
    # hebbian_decay = 1
    # oja_alpha = np.sqrt(model.input_size)
    # oja_alpha = 1

    # Training Loop
    epochs = 3000
    has_boundary = False
    has_hebbian = False
    narrow_factor = 0
    max_narrow_factor = 0.0001
    narrow_up_rate = max_narrow_factor / 1000
    gc_thresh = np.sqrt(input_size) * 0.01
    sc_thresh = np.sqrt(input_size) * 0.01
    hebbian_lr = 0
    max_hebbian_lr = 0.001
    hebbian_up_rate = max_hebbian_lr / 1000
    hebb_alpha = 5.5
    oja_alpha = 12

    losses = []
    weight_sums = []
    weights = []
    gain_changes = []
    shift_changes = []
    epoch_loss = 0

    for epoch in tqdm(range(epochs), position=0, leave=True):

        # shuffle data
        perm_idx = torch.randperm(ndata)
        shuffled_xs = xs[perm_idx]
        shuffled_ys = ys[perm_idx]
        last_epoch_loss = epoch_loss
        epoch_loss = 0

        for x, y in zip(shuffled_xs, shuffled_ys):

            # establish model
            model = SimpleNeuralNetwork(input_size, init_gain, init_shift, init_weight)
            loss = model.train_point(x, y)
            epoch_loss += loss

            # update init gains and shifts
            init_gain = model.gain.detach().numpy()
            init_shift = model.shift.detach().numpy()
            gain_change = np.linalg.norm(init_gain - theo_gain, 2)
            shift_change = np.linalg.norm(init_shift - theo_shift, 2)

            # hebbian learning
            if epoch > 200 and last_epoch_loss < 0.001 and has_hebbian == False:
                print("hebbian start!!!")
                has_hebbian = True
            if has_hebbian and loss < 0.001:
            # if has_hebbian:
                # Update hebbian lr if not max
                if hebbian_lr < max_hebbian_lr:
                    hebbian_lr += hebbian_up_rate/ndata
                # Calculate Hebbian weight updates
                hebbian_update = model.output_activation * (model.input_activation).T
                # # Apply Hebbian updates and normalize
                # model.weights = model.weights + hebbian_lr * hebbian_update
                # model.weights = model.weights / torch.sum(model.weights) * hebb_alpha
                # Regulation term of Oja
                rj_square = (model.output_activation**2).expand(-1, model.input_size)
                oja_regulation = oja_alpha * rj_square * model.weights
                # Oja's rule
                model.weights = model.weights + hebbian_lr * hebbian_update - hebbian_lr * oja_regulation            
            # update init weights
            init_weight = model.weights.detach().numpy()

            # shrink shift and gain to init value
            if epoch > 400 and last_epoch_loss < 0.001 and has_boundary == False:
                # create boundaries
                gain_ub = np.maximum(init_gain, theo_gain)
                gain_lb = np.minimum(init_gain, theo_gain)
                shift_ub = np.maximum(init_shift, theo_shift)
                shift_lb = np.minimum(init_shift, theo_shift)
                has_boundary = True
                print("boundary start!!!")
            if has_boundary and last_epoch_loss < 0.001:
                # update narrow factor if not max
                if narrow_factor < max_narrow_factor:
                    narrow_factor += narrow_up_rate/ndata
                # passively narrow the boundaries
                gain_ub = np.maximum(np.minimum(init_gain, gain_ub), theo_gain)
                gain_lb = np.minimum(np.maximum(init_gain, gain_lb), theo_gain)
                shift_ub = np.maximum(np.minimum(init_shift, shift_ub), theo_shift)
                shift_lb = np.minimum(np.maximum(init_shift, shift_lb), theo_shift)
                # actively narrow the boundaries
                if np.linalg.norm(gain_ub - theo_gain, 2) > gc_thresh:
                    gain_ub -= narrow_factor * (gain_ub - theo_gain)
                if np.linalg.norm(gain_lb - theo_gain, 2) > gc_thresh:
                    gain_lb -= narrow_factor * (gain_lb - theo_gain)
                if np.linalg.norm(shift_ub - theo_shift, 2) > sc_thresh:
                    shift_ub -= narrow_factor * (shift_ub - theo_shift)
                if np.linalg.norm(shift_lb - theo_shift, 2) > sc_thresh:
                    shift_lb -= narrow_factor * (shift_lb - theo_shift)
            # pull gains and shifts back to into boundaries
            if has_boundary:
                init_gain = np.minimum(init_gain, gain_ub)
                init_gain = np.maximum(init_gain, gain_lb)
                init_shift = np.minimum(init_shift, shift_ub)
                init_shift = np.maximum(init_shift, shift_lb)

        # calc epoch loss
        epoch_loss /= ndata

        # print out info
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss},\nGC:{gain_change},SC:{shift_change}, WC:{np.sum(init_weight[:, 0:100])}')
            print(model.gain.detach().numpy()[0:10])

        # record
        losses.append(epoch_loss)
        weight_sums.append(np.sum(init_weight[:, 0:100]))
        gain_changes.append(gain_change)
        shift_changes.append(shift_change)
        weights.append(init_weight)

        # termination
        if epoch > 0.8 * epochs and epoch_loss < 0.001 and gain_change < gc_thresh and shift_change < sc_thresh:
            break

    # true epoch
    epochs = epoch + 1
    print(f"true epochs: {epochs}")

    filename = "abb05_rep.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
        pickle.dump(losses, f)
        pickle.dump(weight_sums, f)
        pickle.dump(gain_changes, f)
        pickle.dump(shift_changes, f)
        pickle.dump(weights, f)