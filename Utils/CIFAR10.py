# ***************************************************************
# Copyright(c) 2019
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>.
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

import numpy as np
import gzip
from PIL import Image
# our lib jittor import
from jittor.dataset.dataset import Dataset
# from jittor.dataset.utils import ensure_dir, download_url_to_local
import jittor as jt
import jittor.transform as trans
import DataExtractor
import matplotlib.pyplot as plt

class CIFAR10(Dataset):
    def __init__(self, data_root, train=True ,download=True, transform=None):
        # if you want to test resnet etc you should set input_channel = 3, because the net set 3 as the input dimensions
        super().__init__()
        self.data_root = data_root
        self.is_train = train
        self.transform = transform
        # if download == True:
        #     self.download_url()

        X_train, y_train, X_val, y_val, X_test, y_test = DataExtractor.get_CIFAR10_data(path_to_cifar_dir=data_root)
        # print("X_train's shape", np.shape(X_train))

        self.cifar10 = {}
        if self.is_train:
            self.cifar10["images"] = X_train
            self.cifar10["labels"] = y_train
        else:
            self.cifar10["images"] = X_val
            self.cifar10["labels"] = y_val
        # print("cifar10 images' shape: ", np.shape(self.cifar10["images"]))
        # print("cifar10 labels' shape: ", np.shape(self.cifar10["labels"]))
        assert(self.cifar10["images"].shape[0] == self.cifar10["labels"].shape[0])
        self.total_len = self.cifar10["images"].shape[0]
        # this function must be called
        self.set_attrs(total_len = self.total_len)

    def __getitem__(self, index):
        # print("Image Process' shape: ", np.shape(self.cifar10['images'][index]))
        # img = Image.fromarray(self.cifar10['images'][index]).convert('RGB')
        # img = Image.fromarray(np.uint8(self.cifar10['images'][index]), mode='RGB')
        img = Image.fromarray(np.array(self.cifar10['images'][index], dtype='uint8'))
        # img = self.cifar10['images'][index]
        # img = Image.fromarray(np.transpose(self.cifar10['images'][index], (1, 2, 0)), mode='RGB')
        # print("img's shape: ", np.shape(img))
        if self.transform:
            img = self.transform(img)
            # print("transformed img's shape: ", np.shape(img))

        # fig = plt.figure()
        # plt.imshow(img)
        # plt.show()
        # print(self.cifar10['labels'][index])
        return trans.to_tensor(img), self.cifar10['labels'][index]

    # def download_url(self):
    #     resources = [
    #         ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    #         ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    #         ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    #         ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    #     ]
    #
    #     for url, md5 in resources:
    #         filename = url.rpartition('/')[2]
    #         download_url_to_local(url, filename, self.data_root, md5)