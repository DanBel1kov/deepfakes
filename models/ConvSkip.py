from torch import nn
import torch.nn.functional as F


class ConvSkip(nn.Module):
    def __init__(self):
        super(ConvSkip, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=3, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(5 * 5 * 128, 1024)
        self.fc2 = nn.Linear(1024, 1)
        # self.fc3 = nn.Linear(30, 1)
        # self.fc2_2 = nn.Linear(1000, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.avgpool(self.conv1(x))))
        x = F.relu(self.bn2(self.pool(self.conv2(x))))
        x = F.relu(self.bn3(self.pool(self.conv3(x))))
        x = F.relu(self.bn3(self.pool(self.conv4(x))))
        # print(x.shape)
        x = x.view(-1, 5 * 5 * 128)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc2(x))
        return x
