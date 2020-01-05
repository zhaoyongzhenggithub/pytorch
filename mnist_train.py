import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from matplotlib import pyplot as plt
from utils import plot_image, plot_curve, one_hot

batch_size = 512

#step1. load dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'image sample')

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        #xw+b
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        #x: [b, 1, 28, 28]
        #h1 = relu(xw1 + b1)
        x = F.relu(self.fc1(x))
        #h2 = relu(h1w2 + b2)
        x = F.relu(self.fc2(x))
        #h3 = h2w3 +　b3
        x = self.fc3(x)

        return x

net = Net()
#[w1, b1, w2, b2, w3, b3]
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train_loss = []

for epoch in range(3):

    for batch_idx, (x, y) in enumerate(train_loader):
