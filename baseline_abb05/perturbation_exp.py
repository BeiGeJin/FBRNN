import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from training_abb05_bphebb import SimpleNeuralNetwork
from tqdm import tqdm

# define the simulator
class PerturbNetwork():
     
    def __init__(self, model_rep, simu_epochs=1500, perturb_start=50, perturb_last=500, perturb_amp=1, only_backprop_epoch=10,
                 backprop_lr=0.2, hebbian_lr=0.0001, hebb_alpha=5.5):
        # init params
        self.simu_epochs = simu_epochs
        self.perturb_start = perturb_start
        self.perturb_last = perturb_last
        self.perturb_amp = perturb_amp
        self.only_backprop_epoch = only_backprop_epoch
        self.bound_start_epoch = 20
        self.backprop_lr = backprop_lr
        self.hebbian_lr = hebbian_lr
        self.hebb_alpha = hebb_alpha

        # some hard defined params
        self.input_size = 230
        self.theo_gain = 3 * np.ones((self.input_size, 1))
        self.theo_shift = 1 * np.ones((self.input_size, 1))

        # init models
        self.init_gain = model_rep.gain.detach().numpy()
        self.init_shift = model_rep.shift.detach().numpy()
        self.init_weight = model_rep.weights.detach().numpy()

    def simulate(self, ndata=200, seed=0, perturb_in_sigmoid=False):
        # set seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # data prep
        xs = torch.linspace(0, 2 * torch.pi, ndata)
        ys = torch.cos(xs)/4 + 0.5
    
        # define noise
        W = np.eye(self.input_size) * 0.001
        self.x_noises = (np.random.multivariate_normal(mean=np.zeros(self.input_size), cov=W, size=self.simu_epochs)).T * 0
        self.x_noises = torch.tensor(self.x_noises, dtype=torch.float32)
        self.x_noises[:,self.perturb_start:self.perturb_start+self.perturb_last] += self.perturb_amp
        self.x_noises[:,0:self.perturb_start] *= 0
        self.x_noises[:,self.perturb_start+self.perturb_last:] *= 0

        # create boundary
        gain_ub = np.maximum(self.init_gain, self.theo_gain)
        gain_lb = np.minimum(self.init_gain, self.theo_gain)
        shift_ub = np.maximum(self.init_shift, self.theo_shift)
        shift_lb = np.minimum(self.init_shift, self.theo_shift)

        # flags
        has_backprop = True  # always true
        has_boundary = False
        has_hebbian = True
        has_perturb = False

        # to record
        simu_losses = []
        gain_changes = []
        shift_changes = []
        simu_weights = []
        simu_gains = []
        simu_shifts = []
        epoch_loss = 0

        for epoch in tqdm(range(self.simu_epochs)):
            if epoch == self.perturb_start:
                has_perturb = True
                has_hebbian = False
                has_boundary = False
                # hebbian_lr = 0
                print("perturbation start!!!")
            if epoch == self.perturb_start + self.perturb_last:
                has_perturb = False
                has_hebbian = False
                has_boundary = False
                # hebbian_lr = 0
                print("perturbation end!!!")

            # shuffle data
            ndata = len(xs)
            perm_idx = torch.randperm(ndata)
            shuffled_xs = xs[perm_idx]
            shuffled_ys = ys[perm_idx]
            last_epoch_loss = epoch_loss
            epoch_loss = 0

            # start hebbian and shrinkage
            if has_perturb and epoch > self.perturb_start + self.only_backprop_epoch and last_epoch_loss < 0.001 and has_hebbian == False:
                has_hebbian = True
                print("perturb learning start!!!")
            if not has_perturb and epoch > self.perturb_start + self.perturb_last + self.only_backprop_epoch and last_epoch_loss < 0.001 and has_hebbian == False:
                has_hebbian = True
                print("origin learning start!!!")
            # if has_perturb and epoch > self.perturb_start + self.only_backprop_epoch and last_epoch_loss < 0.001 and has_boundary == False:
            #     gain_ub = np.maximum(self.init_gain, self.theo_gain)
            #     gain_lb = np.minimum(self.init_gain, self.theo_gain)
            #     shift_ub = np.maximum(self.init_shift, self.theo_shift)
            #     shift_lb = np.minimum(self.init_shift, self.theo_shift)
            #     has_boundary = True
            #     print("perturb boundary created!!!")
            if not has_perturb and epoch > self.perturb_start + self.perturb_last + self.only_backprop_epoch and last_epoch_loss < 0.001 and has_boundary == False:
            # if not has_perturb and epoch > self.perturb_start + self.perturb_last + self.bound_start_epoch and has_boundary == False:
                gain_ub = np.maximum(self.init_gain, self.theo_gain)
                gain_lb = np.minimum(self.init_gain, self.theo_gain)
                shift_ub = np.maximum(self.init_shift, self.theo_shift)
                shift_lb = np.minimum(self.init_shift, self.theo_shift)
                has_boundary = True
                print("origin boundary created!!!")
                
            # # update hebbian learning rate, once per epoch
            # if has_hebbian and hebbian_lr < max_hebbian_lr:
            #     hebbian_lr += hebbian_up_rate
            
            # go through all data
            for x, y in zip(shuffled_xs, shuffled_ys):  
                # establish model
                model = SimpleNeuralNetwork(self.input_size, self.init_gain, self.init_shift, self.init_weight)
                # forward
                inpu_ipl = model.gaussian_rf(x)
                if perturb_in_sigmoid:
                    actv_ipl = model.activation_func(model.gain * (inpu_ipl - model.shift) + (self.x_noises[:, epoch]).reshape(-1, 1))  # inside sigmoid
                else:
                    actv_ipl = model.activation_func(model.gain * (inpu_ipl - model.shift)) + (self.x_noises[:, epoch]).reshape(-1, 1)  # outside sigmoid
                model.input_activation = actv_ipl.clone()
                inpu_opl = torch.matmul(model.weights, actv_ipl)
                actv_opl = model.activation_func(model.gainout * (inpu_opl - model.shiftout))
                model.output_activation = actv_opl.clone()          
                output = actv_opl.squeeze()
                # Calculate loss
                loss_func = nn.MSELoss()
                loss = 0.5 * loss_func(output, y)
                epoch_loss += loss

                # backprop for gains and shifts
                if has_backprop:
                    optimizer = optim.SGD([model.gain, model.shift], lr=self.backprop_lr)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                # update init gains and shifts
                self.init_gain = model.gain.detach().numpy()
                self.init_shift = model.shift.detach().numpy()
                gain_change = np.linalg.norm(self.init_gain - self.theo_gain, 2)
                shift_change = np.linalg.norm(self.init_shift - self.theo_shift, 2)

                # hebbian learning for weights
                if has_hebbian:
                    # Calculate Hebbian weight updates
                    hebbian_update = model.output_activation * (model.input_activation).T
                    # Apply Hebbian updates and normalize
                    model.weights = model.weights + self.hebbian_lr * hebbian_update
                    model.weights = model.weights / torch.sum(model.weights) * self.hebb_alpha
                # update init weights
                self.init_weight = model.weights.detach().numpy()

                # shrink shift and gain to init value
                if has_boundary:
                    # passively narrow the boundaries
                    # if gain_change > 0.1:
                    gain_ub = np.maximum(np.minimum(self.init_gain, gain_ub), self.theo_gain)
                    gain_lb = np.minimum(np.maximum(self.init_gain, gain_lb), self.theo_gain)
                    # if shift_change > 0.1:
                    shift_ub = np.maximum(np.minimum(self.init_shift, shift_ub), self.theo_shift)
                    shift_lb = np.minimum(np.maximum(self.init_shift, shift_lb), self.theo_shift)
                    # pull gains and shifts back to into boundaries
                    self.init_gain = np.minimum(self.init_gain, gain_ub)
                    self.init_gain = np.maximum(self.init_gain, gain_lb)
                    self.init_shift = np.minimum(self.init_shift, shift_ub)
                    self.init_shift = np.maximum(self.init_shift, shift_lb)

            # print losses
            epoch_loss /= ndata
            # if epoch % 50 == 0:
            #     print(f"Epoch: {epoch}, Loss: {epoch_loss}")
            
            # record
            simu_losses.append(epoch_loss.item())
            gain_changes.append(gain_change)
            shift_changes.append(shift_change)
            simu_weights.append(self.init_weight.copy())
            simu_gains.append(self.init_gain.copy())
            simu_shifts.append(self.init_shift.copy())

        return simu_losses, gain_changes, shift_changes, simu_weights, simu_gains, simu_shifts, model


# systematic differentiate the perturbation lasts
if __name__ == "__main__":

    # load the pickle file
    with open('weights_abb05_bphebb.pkl', 'rb') as f:
        model_rep = pickle.load(f)

    # define hyper-parameters
    # iter_num = 10
    iter_num = 3
    simu_epochs = 4000
    # perturb_lasts_exp = np.arange(1,3.1,0.1)
    # perturb_lasts = np.power(10, perturb_lasts_exp).astype(int)
    # perturb_lasts =  np.unique(perturb_lasts.round(-1))
    # perturb_lasts = np.append(perturb_lasts, [1200, 1500, 2000])
    perturb_lasts = [500, 1000, 1500, 2000, 2200, 2500, 2800, 3000, 3500]  # for inside sigmoid
    perturb_amp = 1
    print(perturb_lasts)

    # record
    all_perturb_lasts = []
    all_simu_losses = []
    all_gain_changes = []
    all_shift_changes = []
    all_simu_weights = []
    all_simu_gains = []
    all_simu_shifts = []

    # start
    for i in range(iter_num):
        for perturb_last in perturb_lasts:
            print("Now Iter ...", i)
            print("Now Start ...", perturb_last)
            simulator = PerturbNetwork(model_rep, simu_epochs=simu_epochs, perturb_amp=perturb_amp, perturb_last=perturb_last, only_backprop_epoch=0)
            simu_losses, gain_changes, shift_changes, simu_weights, simu_gains, simu_shifts, model_final = simulator.simulate(ndata=200, seed=i, perturb_in_sigmoid=True)
            all_perturb_lasts.append(perturb_last)
            all_simu_losses.append(simu_losses)
            all_gain_changes.append(gain_changes)
            all_shift_changes.append(shift_changes)
            all_simu_weights.append(simu_weights)
            all_simu_gains.append(simu_gains)
            all_simu_shifts.append(simu_shifts)
    
    # save the results
    filename = "perturbation_exp_result.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(all_perturb_lasts, f)
        pickle.dump(all_simu_losses, f)
        pickle.dump(all_gain_changes, f)
        pickle.dump(all_shift_changes, f)
        pickle.dump(all_simu_weights, f)
        pickle.dump(all_simu_gains, f)
        pickle.dump(all_simu_shifts, f)