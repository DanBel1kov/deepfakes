from torch import nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 1)
        # self.fc3 = nn.Linear(30, 1)
        # self.fc2_2 = nn.Linear(1000, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        skip = x
        x = F.relu(self.bn1(self.conv2(x)))
        x = self.bn1(self.conv2(x))
        x += skip
        x = F.relu(x)
        x = self.avpool(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.pool(self.conv4(x))))

        # print(x.shape)

        x = x.view(-1, 4 * 4 * 128)
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc2(x))
        return x
