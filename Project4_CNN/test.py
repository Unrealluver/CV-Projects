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
from ErrorBarUtil import *
from vgg_16 import VGG_16
from vgg_11 import VGG_11
from mini_vgg import MINI_VGG
from cifar10_fast_model import CIFAR10_FAST_MODEL

jt.flags.use_cuda = 0 # if jt.flags.use_cuda = 1 will use gpu

def train(model, train_loader, optimizer, epoch):
    model.train()
    loss_list = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        outputs = model(inputs)
        # print("outputs & targets' shape: ", np.shape(outputs), np.shape(targets))
        loss = nn.cross_entropy_loss(outputs, targets)
        loss_list.append(loss.data)
        optimizer.step (loss)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data[0]))
            # break
    return loss_list


def val(model, val_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total_acc = 0
    total_num = 0
    acc_list = []
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        pred = np.argmax(outputs.data, axis=1)
        acc = np.sum(targets.data==pred)
        total_acc += acc
        total_num += batch_size
        acc = acc / batch_size
        acc_list.append(acc)
        print('Test Epoch: {} [{}/{} ({:.0f}%)]\tAcc: {:.6f}'.format(epoch, \
                    batch_idx, len(val_loader),100. * float(batch_idx) / len(val_loader), acc))
    print ('Total test acc =', total_acc / total_num)
    return acc_list



'''
lr = 0.1 loss dont descent, too large
lr = 0.03 loss dont descent, too large
lr = 0.01 65 % acc
lr = 0.003 epoch = 5 -> 67.8% acc 
lr = 0.003 adam epoch = 5 -> 62.4% acc
lr = 0.0003 adam epoch = 5 -> 65.7% acc
lr = 0.0001 adam epoch = 5 -> 65.6% acc
lr = 0.0001 adam epoch = 10 tbs128/32 -> 68.8% acc
lr = 0.0001 adam epoch = 10 tbs256/64 -> 68.3% acc
lr = 0.0001 adam epoch = 10 tbs256/64 vgg11 -> 58.9% acc
lr = 0.0003 optimizer=adam epochs=10train_batch_size256test_batch_size64 vgg11 -> 68.9%
lr = 0.0003 adam epoch = 10 tbs256/64 vgg16 -> 62.8%acc
lr = 0.003 adam epoch = 10 tbs256/64 minivgg -> 66.6^
'''
def main ():
    train_batch_size = 64
    test_batch_size = 32
    learning_rate = 0.003
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 10
    loss_matrix = []
    acc_matrix = []
    data_root = "../Utils/cifar-10-batches-py"
    optimizerID = 'adam'
    plt_path = os.path.abspath('.') + "/plt/"

    train_loader = CIFAR10(train=True, data_root=data_root, transform=trans.Resize(32)).set_attrs(batch_size=train_batch_size, shuffle=True)
    # train_loader = CIFAR10()
    val_loader = CIFAR10(train=False, data_root=data_root, transform=trans.Resize(32)).set_attrs(batch_size=test_batch_size, shuffle=False)

    model = CIFAR10_FAST_MODEL()
    if optimizerID == 'adam':
        optimizer = nn.Adam(model.parameters(), learning_rate, weight_decay)
    elif optimizerID == 'SGD':
        optimizer = nn.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    for epoch in range(epochs):
        loss_list = train(model, train_loader, optimizer, epoch)
        loss_matrix.append(loss_list)
        acc_list = val(model, val_loader, epoch)
        acc_matrix.append(acc_list)

    # jittor core var
    # np.save("./lm.npy", np.array(loss_matrix))
    # np.save("./am.npy", np.array(acc_matrix))
    draw_error_bar(loss_matrix, 'mini 2bn' + 'epoch', 'loss', 'err ' + 'vgg' + 'lr=' + learning_rate.__str__()
                   + " optimizer=" + optimizerID + " epochs=" + epochs.__str__()
                   + "tbs" + train_batch_size.__str__()
                   + " " + test_batch_size.__str__()
                   # + "conv 64/128"
                   , save_dir=plt_path)
    draw_error_bar(acc_matrix, 'mini 2bn ' + 'epoch', 'acc', 'acc ' + 'vgg' + 'lr=' + learning_rate.__str__()
                   + " optimizer=" + optimizerID + " epochs=" + epochs.__str__()
                   + "tbs" + train_batch_size.__str__()
                   + " " + test_batch_size.__str__()
                   # + "conv 64/128"
                   , save_dir=plt_path)

if __name__ == '__main__':
    main()