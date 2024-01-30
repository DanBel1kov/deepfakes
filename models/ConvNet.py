from torch import nn
from torch import flatten
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=1, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=3),
            nn.Conv2d(32, 64, kernel_size=5, padding=1, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1600, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = flatten(x, 1)
        x = self.classifier(x)
        return x
