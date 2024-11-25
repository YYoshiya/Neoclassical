import time
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Model(nn.Module):
    def __init__(self, hidden_size=24):
        super(ValueNetwork, self).__init__()
        self.dense1 = nn.Linear(1, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense3 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.dense1(x)
        x = self.Tanh(x)
        x = self.dense2(x)
        x = self.Tanh(x)
        x = self.dense3(x)
        x = self.Tanh(x)
        x = self.output(x)
        return x