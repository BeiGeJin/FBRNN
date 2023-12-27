import numpy as np
from torch.autograd.functional import jacobian
from torch import tensor
import torch

class ExtendedKalmanFilter():
    def __init__(self, f_tensor):
        self.f_tensor = f_tensor  # the function in tensor representation
    
    def f(self, x, u):
        x_tensor = tensor(x)
        u_tensor = tensor(u)
        y0, y1 = self.f_tensor(x_tensor, u_tensor)
        y = np.array([[y0.item()], [y1.item()]])
        return y

    def jacob(self, x, u):
        x_tensor = tensor(x)
        u_tensor = tensor(u)
        dydx0u0, dydx1u1 = jacobian(self.f_tensor,(x_tensor, u_tensor))
        dydx0, dydx1 = dydx0u0[0], dydx1u1[0]
        J = np.column_stack([dydx0.numpy()[0], dydx1.numpy()[0]])
        return J

    