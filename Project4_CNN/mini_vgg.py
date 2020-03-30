import jittor as jt
from jittor import nn, Module
import numpy as np
import sys, os
import random
import math
from jittor import init


class MINI_VGG(nn.Module):
    def __init__(self):
        super(MINI_VGG, self).__init__()
        self.conv1_1 = nn.Conv(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.Pool(kernel_size=2, stride=2, op='maximum')
        self.conv2_1 = nn.Conv(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.Pool(kernel_size=2, stride=2, op='maximum')
        self.relu = nn.Relu()
        self.bn64 = nn.BatchNorm(64)
        self.bn128 = nn.BatchNorm(128)
        self.fc1 = nn.Linear (8192, 1024)
        self.fc2 = nn.Linear (1024, 128)
        self.fc3 = nn.Linear (128, 10)

    def execute(self, image):
        out = nn.relu(self.conv1_1(image))
        out = nn.relu(self.conv1_2(out))
        out = self.bn64(out)
        out = self.pool1(out)
        out = nn.relu(self.conv2_1(out))
        out = nn.relu(self.conv2_2(out))
        out = self.bn128(out)
        out = self.pool2(out)

        x = jt.reshape(out, [out.shape[0], -1])
        # print("x's shape: ", np.shape(x))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x