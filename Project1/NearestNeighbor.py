import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X, data_set_proportion = 1):
        num_test = int(X.shape[0] * data_set_proportion)
        Ypred = np.zeros(num_test)
        # Ypred = np.zeros(num_test, dtype = self.ytr[0].dtype)

        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis = 1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]

        return Ypred