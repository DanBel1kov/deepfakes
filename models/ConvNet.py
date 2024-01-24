from torch import nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.skip_connection1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(4 * 4 * 128, 512)
        self.fc2 = nn.Linear(512, 1)
        # self.fc3 = nn.Linear(30, 1)
        # self.fc2_2 = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.bn1(self.pool(F.relu(self.conv1(x))))
        x = self.bn2(self.pool(F.relu(self.conv2(x))))
        x = (self.pool(F.relu(self.conv3(x))))
        x = self.bn2(self.pool(F.relu(self.conv4(x))))
        x = self.bn3(self.pool(F.relu(self.conv5(x))))
        # print(x.shape)
        x = x.view(-1, 4 * 4 * 128)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc2(x))
        return x
