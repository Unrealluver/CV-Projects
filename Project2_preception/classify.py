# %%

from __future__ import print_function
import random
import numpy as np
import matplotlib.pyplot as plt
from softmax import *
from DataExtractor import *
from hog import *

# %%

# 读取数据
X_train, y_train, X_test, y_test = get_all_data()
# ifhog
# X_train, y_train, X_test, y_test = get_hog_data(X_train, y_train, X_test, y_test, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(4, 4))

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# 使用softmax 分类
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_linear_data()

results = {}
best_val = -1
best_softmax = None
learning_rates = [1e-7, 5e-7]
regularization_strengths = [2.5e4, 5e4]

for lr in learning_rates:
    for rs in regularization_strengths:
        softmaxClf = softmax()
        loss_hist = softmaxClf.train(X=X_train, y=y_train, learning_rate=lr, reg=rs,
                                     num_iters=1000, verbose=False)

        y_val_pred = softmaxClf.predict(X_val)
        y_train_pred = softmaxClf.predict(X_train)
        train_acc = np.mean(y_train == y_train_pred)
        val_acc = np.mean(y_val == y_val_pred)
        results[(lr, rs)] = (train_acc, val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_softmax = softmaxClf
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
        lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)



