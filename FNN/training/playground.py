import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

m1 = [3,3,3]
m2 = [5,5,5]

gain = torch.tensor([m1,m2], dtype=torch.float32, requires_grad=True)

print(gain[0][0])