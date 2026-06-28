import torch
from torch import nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class ALL_CNN_C(BasicModule):

    def __init__(self, num_classes=100, in_channels=4):
        super(ALL_CNN_C, self).__init__()

        self.model_name = 'ALL_CNN_C'

        self.conv1 = nn.Conv2d(in_channels, 96, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(96, 96, 3, stride=2, padding=1, bias=False)
        self.dp1 = nn.Dropout(p=0.5)

        self.conv4 = nn.Conv2d(96, 192, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1, bias=False)
        self.conv6 = nn.Conv2d(192, 192, 3, stride=2, padding=1, bias=False)
        self.dp2 = nn.Dropout(p=0.5)

        self.conv7 = nn.Conv2d(192, 192, 3, padding=1, bias=False)
        self.conv8 = nn.Conv2d(192, 192, 1, bias=False)
        self.conv9 = nn.Conv2d(192, num_classes, 1)

        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(96)
        self.bn3 = nn.BatchNorm2d(96)
        self.bn4 = nn.BatchNorm2d(192)
        self.bn5 = nn.BatchNorm2d(192)
        self.bn6 = nn.BatchNorm2d(192)
        self.bn7 = nn.BatchNorm2d(192)
        self.bn8 = nn.BatchNorm2d(192)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dp1(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dp2(x)

        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.conv9(x)

        x = self.avg(x)
        return torch.flatten(x, 1)
