import jittor as jt
from jittor import nn, Module
import numpy as np
import sys, os
import random
import math
from jittor import init


class CIFAR10_FAST_MODEL(nn.Module):
    def __init__(self):
        super(CIFAR10_FAST_MODEL, self).__init__()
        self.conv1_1 = nn.Conv(3, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.Pool(kernel_size=2, stride=2, op='maximum')
        self.dropout1 = nn.Dropout(p=0.2)
        self.bn32 = nn.BatchNorm(32)
        self.conv2_1 = nn.Conv(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.Pool(kernel_size=2, stride=2, op='maximum')
        self.dropout2 = nn.Dropout(p=0.3)
        self.bn64 = nn.BatchNorm(64)
        self.conv3_1 = nn.Conv(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.Pool(kernel_size=2, stride=2, op='maximum')
        self.dropout3 = nn.Dropout(p=0.4)
        self.bn128 = nn.BatchNorm(128)
        self.relu = nn.Relu()
        self.fc1 = nn.Linear (2048, 10)


    def execute(self, image):
        out = nn.relu(self.conv1_1(image))
        out = nn.relu(self.conv1_2(out))
        out = self.bn32(out)
        out = self.pool1(out)
        out = self.dropout1(out)
        out = nn.relu(self.conv2_1(out))
        out = nn.relu(self.conv2_2(out))
        out = self.bn64(out)
        out = self.pool2(out)
        out = self.dropout2(out)
        out = nn.relu(self.conv3_1(out))
        out = nn.relu(self.conv3_2(out))
        out = self.bn128(out)
        out = self.pool3(out)
        out = self.dropout3(out)

        x = jt.reshape(out, [out.shape[0], -1])
        # print("x's shape: ", np.shape(x))
        x = self.fc1(x)
        return x