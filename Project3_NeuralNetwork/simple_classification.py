__author__ = 'm.bashari'
import numpy as np
from sklearn import datasets, linear_model
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from NeuralNetwork import *
import math


def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y


def visualize(X, y, clf):
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    plot_decision_boundary(lambda x: clf.predict(x), X, y)
    plt.title("Logistic Regression")


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    print("Z's len : ", len(Z))
    print("Z's contents : ", Z)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


def classify(X, y):
    # clf = linear_model.LogisticRegressionCV()
    clf = KNeighborsClassifier(metric='cosine')
    # clf = nn_classifier()
    clf.fit(X, y)
    # clf.predict(X, y)
    return clf


class nn_classifier(object):

    # get the y matched 9 kinds of index(from 0 to 8)
    # train_y -= np.ones(len(train_y), dtype=int)
    # test_y -= np.ones(len(test_y), dtype=int)

    def __init__(self):
        # big lr selected warning: NaN -> exp overflow
        self.lr = 0.5
        self.regu_rate = 0.001
        self.max_iter = 300
        self.loss_old = 9999999999999

        self.W1 = np.random.randn(2, 8) / math.sqrt(2)
        self.b1 = np.zeros(8)
        self.W2 = np.random.randn(8, 4) / math.sqrt(8)
        self.b2 = np.zeros(4)
        self.W3 = np.random.randn(4, 2) / math.sqrt(4)
        self.b3 = np.zeros(2)

        self.fc1 = FC(self.W1, self.b1, self.lr, self.regu_rate)
        self.relu1 = Relu()
        self.fc2 = FC(self.W2, self.b2, self.lr, self.regu_rate)
        self.relu2 = Relu()
        self.fc3 = FC(self.W3, self.b3, self.lr, self.regu_rate)
        self.cross_entropy = SparseSoftmaxCrossEntropy()

    def fit(self, train_X, train_y):
        for i in range(self.max_iter):
            h1 = self.fc1.forward(train_X)
            h2 = self.relu1.forward(h1)
            h3 = self.fc2.forward(h2)
            h4 = self.relu2.forward(h3)
            h5 = self.fc3.forward(h4)
            # print("h5's contents: ", h5)
            loss = self.cross_entropy.forward(h5, train_y)
            if self.loss_old < loss:
                break
            loss_old = loss
            print("iter: {}, loss：{}".format(i + 1, loss))

            grad_h5 = self.cross_entropy.backprop()
            grad_h4 = self.fc3.backprop(grad_h5)
            grad_h3 = self.relu2.backprop(grad_h4)
            grad_h2 = self.fc2.backprop(grad_h3)
            grad_h1 = self.relu1.backprop(grad_h2)
            grad_X = self.fc1.backprop(grad_h1)

            self.fc3.update()
            self.fc2.update()
            self.fc1.update()

    def predict(self, test_X, test_y):
        valid_h1 = self.fc1.forward(test_X)
        valid_h2 = self.relu1.forward(valid_h1)
        valid_h3 = self.fc2.forward(valid_h2)
        valid_h4 = self.relu1.forward(valid_h3)
        valid_h5 = self.fc3.forward(valid_h4)
        valid_predict = np.argmax(valid_h5, 1)

        valid_acc = np.mean(valid_predict == test_y)
        print('acc: ', valid_acc)


def main():
    X, y = generate_data()
    X = np.array(X)
    y = np.array(y)
    print("X's shape is : ", X.shape)
    print("y's shape is : ", y.shape)
    # visualize(X, y)
    clf = classify(X, y)
    visualize(X, y, clf)


if __name__ == "__main__":
    main()
