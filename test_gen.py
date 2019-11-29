import torch
import torch.nn as nn
import numpy as np
from hologan.generator import Generator

G = Generator().float()

z = np.random.random(size=(2, 128))
pose = np.random.random(size=(2, 3, 4))

t = G.forward(torch.tensor(z).float(), torch.tensor(pose).float())

print(t)
print(t.shape)
