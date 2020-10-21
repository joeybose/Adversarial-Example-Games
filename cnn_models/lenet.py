'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, nc=3, h=32, w=32):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(nc, 6, 5)
        h, w       = round((h-4)/2), round((w-4)/2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        h, w       = round((h-4)/2), round((w-4)/2)
        self.fc1   = nn.Linear(16*h*w, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class MadryLeNet(nn.Module):
    def __init__(self, nc=1, h=28, w=28):
        super(MadryLeNet, self).__init__()
        self.conv1 = nn.Conv2d(nc, 32, 5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.fc1   = nn.Linear(7*7*64, 1024)
        self.fc2   = nn.Linear(1024, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2, padding=1)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2, padding=1)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
