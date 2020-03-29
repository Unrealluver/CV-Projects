import jittor as jt
from jittor import nn, Module
import numpy as np
import sys, os
import random
import math
from jittor import init


class VGG_16(nn.Module):
    def __init__(self):
        super(VGG_16, self).__init__()
        self.conv1_1 = nn.Conv(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.Pool(kernel_size=2, stride=2, op='maximum')
        self.conv2_1 = nn.Conv(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.Pool(kernel_size=2, stride=2, op='maximum')
        self.conv3_1 = nn.Conv(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.Pool(kernel_size=2, stride=2, ceil_mode=True, op='maximum')
        self.conv4_1 = nn.Conv(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.Pool(kernel_size=2, stride=2, op='maximum')
        self.conv5_1 = nn.Conv(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.Pool(kernel_size=3, stride=1, padding=1, op='maximum')
        self.conv6 = nn.Conv(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv(1024, 1024, kernel_size=1)
        self.relu = nn.Relu()
        self.fc1 = nn.Linear (4096, 1024)
        self.fc2 = nn.Linear (1024, 256)
        self.fc3 = nn.Linear (256, 10)

    def execute(self, image):
        out = nn.relu(self.conv1_1(image))
        out = nn.relu(self.conv1_2(out))
        out = self.pool1(out)
        out = nn.relu(self.conv2_1(out))
        out = nn.relu(self.conv2_2(out))
        out = self.pool2(out)
        out = nn.relu(self.conv3_1(out))
        out = nn.relu(self.conv3_2(out))
        out = nn.relu(self.conv3_3(out))
        out = self.pool3(out)
        out = nn.relu(self.conv4_1(out))
        out = nn.relu(self.conv4_2(out))
        out = nn.relu(self.conv4_3(out))
        conv4_3_feats = out
        out = self.pool4(out)
        out = nn.relu(self.conv5_1(out))
        out = nn.relu(self.conv5_2(out))
        out = nn.relu(self.conv5_3(out))
        out = self.pool5(out)
        out = nn.relu(self.conv6(out))
        conv7_feats = nn.relu(self.conv7(out))

        x = jt.reshape(out, [out.shape[0], -1])
        # print("x's shape: ", np.shape(x))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x