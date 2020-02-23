import numpy as np
from sklearn import neighbors, datasets
from DataExtractor import *

n_neighbors = 1
metric = 'cosine'

train_X, train_y, test_X, test_y = get_all_data()

for weights in ['uniform', 'distance']:
    knn = neighbors.KNeighborsClassifier(n_neighbors, weights=weights, metric=metric)
    knn.fit(train_X, train_y)
    prediction = knn.predict(test_X)

    result = prediction - test_y
    true = 0

    for i in range(len(result)):
        if result[i] == 0:
            true += 1

    print('final acc of KNN classifier with cosine distance metric, k=1 and wights = '
          + weights + 'is: ' + '%.2f%%' % (100 * (true / len(result))))

