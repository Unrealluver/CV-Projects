
from collections import Counter
from DistanceMetric import *

class KNearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y, k=1):
        self.k = k
        self.Xtr = X
        self.ytr = y

    def predict(self, X, metric='cosine'):
        Ypred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            if metric == 'cosine':
                distances = cosine(X[i,:], self.Xtr)
            elif metric == 'chebyshev':
                distances = chebyshev(X[i, :], self.Xtr)
            elif metric == 'manhattan':
                distances = manhattan(X[i, :], self.Xtr)
            elif metric == 'euclidean':
                distances = euclidean(X[i, :], self.Xtr)
            else:
                print("Error distance metric!")
                return []
            nearest = np.argsort(distances)
            top_K_y=[self.ytr[i] for i in nearest[:self.k]]
            votes = Counter(top_K_y)
            Ypred[i] = votes.most_common(1)[0][0]
        return Ypred