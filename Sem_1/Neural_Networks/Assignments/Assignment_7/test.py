import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import optimizer
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from random import shuffle

t = torch.tensor(5)

print(t, t.unsqueeze(0))