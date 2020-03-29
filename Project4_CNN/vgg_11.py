import jittor as jt
from jittor import nn, Module
import numpy as np
import sys, os
import random
import math
from jittor import init

class VGG_11 (Module):
    def __init__ (self):
        super (VGG_11, self).__init__()
        self.conv1 = nn.Conv (3, 64, 3, padding=1) # no padding
        self.conv2 = nn.Conv (64, 128, 3, padding=1)
        self.conv3_1 = nn.Conv(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv(256, 256, 3, padding=1)
        self.conv4_1 = nn.Conv(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv(512, 512, 3, padding=1)
        self.conv5_1 = nn.Conv(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv(512, 512, 3, padding=1)
        self.bn = nn.BatchNorm(512)

        self.max_pool_1 = nn.Pool (2, 2)
        self.max_pool_2 = nn.Pool (2, 2)
        self.max_pool_3 = nn.Pool (2, 2)
        self.max_pool_4 = nn.Pool (2, 2)
        self.max_pool_5 = nn.Pool (2, 2)
        self.relu = nn.Relu()
        self.fc1 = nn.Linear (512, 100)
        self.fc2 = nn.Linear (100, 40)
        self.fc3 = nn.Linear (40, 10)
    def execute (self, x) :
        x = self.conv1 (x)
        x = self.relu (x)
        x = self.max_pool_1(x)

        x = self.conv2 (x)
        x = self.relu (x)
        x = self.max_pool_2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.max_pool_3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = self.max_pool_4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool_5(x)

        x = jt.reshape (x, [x.shape[0], -1])
        # print("x's shape: ", np.shape(x))
        x = self.fc1 (x)
        x = self.relu(x)
        x = self.fc2 (x)
        x = self.relu(x)
        x = self.fc3(x)
        return x