from sklearn.datasets import load_iris
from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from DataExtractor import *

train_X, train_y, test_X, test_y = get_all_data()
data_set_proportion = 0.01
k_range = range(1, 101)
k_acc = []
#循环，取k=1到k=31，查看误差效果
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)

    #cv参数决定数据集划分比例，这里是按照4:1划分训练集和测试集
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

