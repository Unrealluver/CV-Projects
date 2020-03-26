import jittor as jt
from jittor import nn, Module
import numpy as np
import sys, os
import random
import math
from jittor import init
from model import Model
import jittor.transform as trans
from CIFAR10 import *

jt.flags.use_cuda = 0 # if jt.flags.use_cuda = 1 will use gpu

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        outputs = model(inputs)
        loss = nn.cross_entropy_loss(outputs, targets)
        optimizer.step (loss)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data[0]))


def val(model, val_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total_acc = 0
    total_num = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        pred = np.argmax(outputs.data, axis=1)
        acc = np.sum(targets.data==pred)
        total_acc += acc
        total_num += batch_size
        acc = acc / batch_size
        print('Test Epoch: {} [{}/{} ({:.0f}%)]\tAcc: {:.6f}'.format(epoch, \
                    batch_idx, len(val_loader),100. * float(batch_idx) / len(val_loader), acc))
    print ('Total test acc =', total_acc / total_num)



def main ():
    batch_size = 32
    learning_rate = 1e-6
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 5
    data_root = "../Utils/cifar-10-batches-py"

    train_loader = CIFAR10(train=True, data_root=data_root, transform=trans.Resize(32)).set_attrs(batch_size=batch_size, shuffle=True)
    # train_loader = CIFAR10()
    val_loader = CIFAR10(train=False, data_root=data_root, transform=trans.Resize(32)).set_attrs(batch_size=8, shuffle=False)

    model = Model ()
    optimizer = nn.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    for epoch in range(epochs):
        train(model, train_loader, optimizer, epoch)
        val(model, val_loader, epoch)

if __name__ == '__main__':
    main()