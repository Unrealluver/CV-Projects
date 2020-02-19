from sklearn.datasets import load_iris
from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from DataExtractor import *


train_X, train_y, test_X, test_y = get_all_data()
data_set_proportion = 1
k_range = range(1, 21)
k_acc = []


for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')

    scores = cross_val_score(knn, train_X[:int(data_set_proportion * train_X.shape[0]), :],
                             train_y[:int(data_set_proportion * train_y.shape[0])],
                             cv=5, scoring='accuracy')
    k_acc.append(scores.mean())
    print("k for " + k.__str__() + "has been tested, its acc is " + '%.2f%%'%(100*k_acc[k - 1]))

#画图，x轴为k值，y值为误差值
plt.plot(k_range, k_acc)
plt.xlabel('Value of K for KNN')
plt.ylabel('Acc')
plt.show()

