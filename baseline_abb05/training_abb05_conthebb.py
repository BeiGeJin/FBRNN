# repitition of abb05, bp on gains and shifts, and then do hebbian learning to transfer learning to weights
# bp first, then gradually turn on hebbian learning and narrowing
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

    # training loop
    epochs = 5000
    gc_thresh = np.sqrt(input_size) * 0.01
    sc_thresh = np.sqrt(input_size) * 0.01
    narrow_factor = 0
    max_narrow_factor = 0.001
    narrow_up_rate = max_narrow_factor / 1000
    hebbian_lr = 0
    max_hebbian_lr = 0.00001  # 0.001
    hebbian_up_rate = max_hebbian_lr / 1000
    oja_alpha = 12
    has_backprop = True
    has_boundary = False
    has_hebbian = False

    losses = []
    gain_changes = []
    shift_changes = []
    weight_sums = []
    saved_epoch = []
    all_weights = []
    epoch_loss = 0

    for epoch in tqdm(range(epochs), position=0, leave=True):

        # shuffle data
        perm_idx = torch.randperm(ndata)
        shuffled_xs = xs[perm_idx]
        shuffled_ys = ys[perm_idx]
        last_epoch_loss = epoch_loss
        epoch_loss = 0

        # update hebbian learning rate, once per epoch
        if has_hebbian and hebbian_lr < max_hebbian_lr:
            hebbian_lr += hebbian_up_rate
    
        if has_boundary:
            # update narrow factor, once per epoch
            if narrow_factor < max_narrow_factor:
                narrow_factor += narrow_up_rate
            # actively narrow the boundaries
            if np.linalg.norm(gain_ub - theo_gain, 2) > gc_thresh:
                gain_ub -= narrow_factor * (gain_ub - theo_gain)
            if np.linalg.norm(gain_lb - theo_gain, 2) > gc_thresh:
                gain_lb -= narrow_factor * (gain_lb - theo_gain)
            if np.linalg.norm(shift_ub - theo_shift, 2) > sc_thresh:
                shift_ub -= narrow_factor * (shift_ub - theo_shift)
            if np.linalg.norm(shift_lb - theo_shift, 2) > sc_thresh:
                shift_lb -= narrow_factor * (shift_lb - theo_shift)
            
            # if gain_change < gc_thresh and shift_change < sc_thresh:
            #     break

        for x, y in zip(shuffled_xs, shuffled_ys):
            
            # establish model
            model = SimpleNeuralNetwork(input_size, init_gain, init_shift, init_weight)

            # forward
            actv_opl = model(x)
            output = actv_opl.squeeze()
            loss_func = nn.MSELoss()
            loss = 0.5 * loss_func(output, y)
            epoch_loss += loss.item()

            # backprop
            if has_backprop:
                optimizer = optim.SGD([model.gain, model.shift], lr=0.2)
                loss.backward()
                optimizer.step()

            # update init gains and shifts
            init_gain = model.gain.detach().numpy()
            init_shift = model.shift.detach().numpy()
            gain_change = np.linalg.norm(init_gain - theo_gain, 2)
            shift_change = np.linalg.norm(init_shift - theo_shift, 2)

            # start hebbian and shrinkage
            if epoch > 10 and last_epoch_loss < 0.001 and has_hebbian == False:
                has_hebbian = True
                has_backprop = False
            if epoch > 10 and last_epoch_loss < 0.001 and has_boundary == False:
                # create boundaries
                gain_ub = np.maximum(init_gain, theo_gain)
                gain_lb = np.minimum(init_gain, theo_gain)
                shift_ub = np.maximum(init_shift, theo_shift)
                shift_lb = np.minimum(init_shift, theo_shift)
                has_boundary = True
                print("learning start!!!")

            # update weights
            if has_hebbian:
                # Calculate Hebbian weight updates
                hebbian_update = model.output_activation * (model.input_activation).T
                # Regulation term of Oja
                rj_square = (model.output_activation**2).expand(-1, model.input_size)
                oja_regulation = oja_alpha * rj_square * model.weights
                # Oja's rule
                model.weights = model.weights + hebbian_lr * hebbian_update - hebbian_lr * oja_regulation            
                # update init weights
            init_weight = model.weights.detach().numpy()

            # shrink shift and gain to init value
            if has_boundary:
                # passively narrow the boundaries
                gain_ub = np.maximum(np.minimum(init_gain, gain_ub), theo_gain)
                gain_lb = np.minimum(np.maximum(init_gain, gain_lb), theo_gain)
                shift_ub = np.maximum(np.minimum(init_shift, shift_ub), theo_shift)
                shift_lb = np.minimum(np.maximum(init_shift, shift_lb), theo_shift)
                # pull gains and shifts back to into boundaries
                init_gain = np.minimum(init_gain, gain_ub)
                init_gain = np.maximum(init_gain, gain_lb)
                init_shift = np.minimum(init_shift, shift_ub)
                init_shift = np.maximum(init_shift, shift_lb)

        # print losses
        epoch_loss /= ndata
        if epoch % 20 == 0:
            print(f"Epoch: {epoch}, Loss: {epoch_loss}")
            saved_epoch.append(epoch)
            all_weights.append(init_weight)

        # record
        losses.append(epoch_loss)
        weight_sums.append(np.sum(init_weight))
        gain_changes.append(gain_change)
        shift_changes.append(shift_change)

        # # termination
        # if epoch > 0.8 * epochs and epoch_loss < 0.001 and gain_change < gc_thresh and shift_change < sc_thresh:
        #     break

    # true epoch
    epochs = epoch + 1
    print(f"true epochs: {epochs}")

    filename = "weights_abb05_conthebb.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
        pickle.dump(losses, f)
        pickle.dump(weight_sums, f)
        pickle.dump(gain_changes, f)
        pickle.dump(shift_changes, f)
        pickle.dump(saved_epoch, f)
        pickle.dump(all_weights, f)