import torch
import torch.nn as nn
import numpy as np
from hologan.discriminator import Discriminator

D = Discriminator(128).float()

x = np.ones(shape=(10, 3, 128, 128))

t = D.forward(torch.tensor(x).float())

print(t)
