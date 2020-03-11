from DataExtractor import *
from NeuralNetwork import *
from FigureDrawer import *
import numpy as np
from hog import *
import math
import os

plt_path = os.path.abspath('.') + "/plt/"
print('plt_path : ', plt_path)
# origin data may lead to nearly random effect,
# gradient is hard to descent, very hard to choose lr.
# train_X, train_y, test_X, test_y = get_all_data()
train_X, train_y, val_X, val_y, test_X, test_y, dev_X, dev_y = get_normalized_data()
# check the training set.
# print(train_X)

# get the y matched 9 kinds of index(from 0 to 8)
train_y -= np.ones(len(train_y), dtype=int)
test_y -= np.ones(len(test_y), dtype=int)

# get one hot y_label
# train_y = get_one_hot_label(train_y, 9)
# test_y = get_one_hot_label(test_y, 9)

# get hoged data
# train_X, train_y, test_X, test_y = get_hog_data(train_X, train_y, test_X, test_y, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(4, 4))

s_line = int(0.7 * train_X.shape[0])
valid_X = train_X[s_line:]
valid_y = train_y[s_line:]

train_X = train_X[:s_line]
train_y = train_y[:s_line]

image_size = train_X.shape[1]
print("image size is : ", image_size)

W1 = np.random.randn(image_size, int(image_size / 3)) / math.sqrt(image_size)
b1 = np.zeros(int(image_size / 3))
W2 = np.random.randn(int(image_size / 3), int(image_size / 3 / 32)) / math.sqrt(int(image_size / 3))
b2 = np.zeros(int(image_size / 3 / 32))
W3 = np.random.randn(int(image_size / 3 / 32), 9) / math.sqrt(int(image_size / 3 / 32))
b3 = np.zeros(9)

'''
lr0.00005 + 0.99decay + 300epochs -> acc15.4% 
lr0.00005 + 300epochs -> acc17.3%
lr0.05 + 300epochs + normalize to (-1, 1) -> acc40.9%
lr0.5 + 300epochs + normalize to (-1, 1) -> acc 40.2%
lr0.5 + 1000epochs + normalize to (-1, 1) -> acc 41.3%
'''
# big lr selected warning: NaN -> exp overflow
lr = 0.5
lr_decay = 0.999
regu_rate = 0.001
max_iter = 300
loss_old = 9999999999999
loss_history = []

fc1 = FC(W1, b1, lr, regu_rate)
relu1 = Relu()
fc2 = FC(W2, b2, lr, regu_rate)
relu2 = Relu()
fc3 = FC(W3, b3, lr, regu_rate)
cross_entropy = SparseSoftmaxCrossEntropy()

for i in range(max_iter):
    h1 = fc1.forward(train_X)
    h2 = relu1.forward(h1)
    h3 = fc2.forward(h2)
    h4 = relu2.forward(h3)
    h5 = fc3.forward(h4)
    # print("h5's contents: ", h5)
    loss = cross_entropy.forward(h5, train_y)
    loss_history.append(loss)
    # update lr to control the direction
    if loss_old < loss and max_iter > 200:
        fc1.update_lr(lr_decay)
        fc2.update_lr(lr_decay)
        fc3.update_lr(lr_decay)
        print("lr changed to : ", str(fc1.get_lr()))
    loss_old = loss
    print("iter: {}, loss：{}".format(i + 1, loss))

    grad_h5 = cross_entropy.backprop()
    grad_h4 = fc3.backprop(grad_h5)
    grad_h3 = relu2.backprop(grad_h4)
    grad_h2 = fc2.backprop(grad_h3)
    grad_h1 = relu1.backprop(grad_h2)
    grad_X = fc1.backprop(grad_h1)

    fc3.update()
    fc2.update()
    fc1.update()

valid_h1 = fc1.forward(valid_X)
valid_h2 = relu1.forward(valid_h1)
valid_h3 = fc2.forward(valid_h2)
valid_h4 = relu1.forward(valid_h3)
valid_h5 = fc3.forward(valid_h4)
valid_predict = np.argmax(valid_h5, 1)

valid_acc = np.mean(valid_predict == valid_y)
print('acc: ', valid_acc.__str__())

draw_figure(range(1, max_iter + 1), loss_history, 'iter', 'loss', 'lr' + lr.__str__(), save_dir=plt_path)

