from DataExtractor import *
from NeuralNetwork import *
import numpy as np
from hog import *
import math

train_X, train_y, test_X, test_y = get_all_data()

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

# big lr selected warning: NaN -> exp overflow
lr = 0.00000005
regu_rate = 0.001
max_iter = 300
loss_old = 9999999999999

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
    if loss_old < loss:
        break
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
print('acc: ', valid_acc)
