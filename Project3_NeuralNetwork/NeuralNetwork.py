import numpy as np
from math import *

class FC:
    def __init__(self, W, b, lr, regu_rate, optimizer='SGD'):
        self.W = W.copy()
        self.b = b.copy()
        self.lr = lr
        self.regu_rate = regu_rate
        if optimizer == 'SGD':
            self.optimizer = SGD()
        elif optimizer == 'Momentum':
            self.optimizer = Momentum()
        elif optimizer == 'AdaGrad':
            self.optimizer = AdaGrad()

    def set_lr(self, lr):
        self.lr = lr

    def update_lr(self, proportion):
        self.lr = proportion * self.lr

    def get_lr(self):
        return self.lr

    def forward(self, X):
        self.X = X.copy()
        return self.X.dot(self.W) + self.b

    def backprop(self, back_grad):
        self.grad_W = self.X.T.dot(back_grad)
        # 1 * k
        self.grad_b = np.ones(self.X.shape[0]).dot(back_grad)
        self.grad = back_grad.dot(self.W.T)
        return self.grad

    def update(self):
        # self.W -= self.lr * (self.grad_W + self.regu_rate * self.W)
        self.W = self.optimizer.update(weights=self.W, grads=self.grad_W, lr=self.lr, regu_rate=self.regu_rate)
        self.b -= self.lr * self.grad_b
        # self.lr *= 0.99


class Relu:
    def forward(self, X):
        self.X = X.copy()
        return np.maximum(X, 0)

    def backprop(self, back_grad):
        grad = back_grad.copy()
        grad[self.X < 0] = 0
        return grad

class Sigmoid:
    def forward(self, X):
        out = sigmoid(X)
        self.out = out
        return out

    def backprop(self, back_grad):
        dx = back_grad * (1.0 - self.out) * self.out
        return dx


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


class SparseSoftmaxCrossEntropy:
    def forward(self, X, y):
        self.X = X.copy()
        # print("X's shape: ", np.shape(self.X))
        self.y = y.copy()
        # print("y's shape: ", np.shape(self.y))
        denom = np.sum(np.exp(self.X), axis=1).reshape([-1, 1])
        self.softmax = np.exp(X) / denom
        # print("softmax's shape: ", np.shape(self.softmax))
        cross_entropy = np.mean(-np.log(self.softmax[range(self.X.shape[0]), self.y]))
        # print("cross_entropy's shape: ", np.shape(cross_entropy))
        return cross_entropy

    def backprop(self):
        m, n = self.X.shape
        activation_mat = np.zeros([m, n])
        activation_mat[range(m), self.y] = 1
        grad = (self.softmax - activation_mat) / m
        return grad

class SGD:
    def update(self, weights, grads, lr, regu_rate):
        weights -= lr * (grads + regu_rate * weights)
        return weights

class Momentum:
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.v = None

    def update(self, weights, grads, lr, regu_rate):
        if self.v is None:
            self.v = np.zeros_like(weights)
        self.v = self.momentum * self.v - lr * grads
        # print("grads' shape: ", np.shape(grads))
        # print("weights' shape: ", np.shape(weights))
        # print("v's shape: ", np.shape(self.v))
        weights += self.v
        return weights

class AdaGrad:
    def __init__(self):
        self.h = None

    def update(self, weights, grads, lr, regu_rate):
        if self.h is None:
            self.h = np.zeros_like(weights)
        self.h = grads * grads
        weights -= lr * grads / (np.sqrt(self.h) + 1e-7)
        return weights


class Dropout:
    # http://arxiv.org/abs/1207.0580
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    # http://arxiv.org/abs/1502.03167

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        self.running_mean = running_mean
        self.running_var = running_var

        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx
