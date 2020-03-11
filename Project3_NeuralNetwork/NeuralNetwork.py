import numpy as np
from math import *

class FC:
    def __init__(self, W, b, lr, regu_rate):
        self.W = W.copy()
        self.b = b.copy()
        self.lr = lr
        self.regu_rate = regu_rate

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
        self.W -= self.lr * (self.grad_W + self.regu_rate * self.W)
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
        return cross_entropy

    def backprop(self):
        m, n = self.X.shape
        activation_mat = np.zeros([m, n])
        activation_mat[range(m), self.y] = 1
        grad = (self.softmax - activation_mat) / m
        return grad