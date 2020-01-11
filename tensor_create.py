import numpy as np
import torch

# Import from numpy
a = np.array([2, 3.3])
print(torch.from_numpy(a))

a = np.ones([2, 3])
print(torch.from_numpy(a))

print("---------------------------------------------------------------------------")

# Import from list
print(torch.tensor([2, 3.2]))
print(torch.FloatTensor([2, 3.2]))
print(torch.tensor([
    [2., 3.2],
    [1., 22.3]
]))