from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from DataExtractor import *
from time import *
from hog import *
import os


def draw_figure(x, y, x_label, y_label, title, save_dir=os.path.split(os.path.realpath(__file__))[0] + "/plt/"):
    plt.grid(True, linestyle='--')
    my_x_ticks = np.arange(0, len(x) + 1, 1)
    plt.xticks(my_x_ticks)
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig((save_dir + title + ".png"), bbox_inches="tight")
    plt.show()


train_X, train_y, test_X, test_y = get_all_data()
train_X, train_y, test_X, test_y = get_hog_data(train_X, train_y, test_X, test_y)
metric_list = ["cosine", "euclidean", "manhattan", "chebyshev"]
data_set_proportion = 0.1
k_range_toplimit = 21
k_range = range(1, k_range_toplimit)
k_acc = []
k_acc_total = []

validation_time = []
validation_time_total = []

for metric in metric_list:
    k_acc = []
    validation_time = []
    for k in k_range:
        print("metric: " + metric + " is testing...")
        start = time()
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)

        scores = cross_val_score(knn, train_X[:int(data_set_proportion * train_X.shape[0]), :],
                                 train_y[:int(data_set_proportion * train_y.shape[0])],
                                 cv=5, scoring='accuracy')
        k_acc.append(scores.mean())
        end = time()
        time_cost = end - start
        print("k for " + k.__str__() + " has been tested.")
        print("its acc is " + '%.2f%%' % (100 * k_acc[k - 1]))
        print("its time cost is", time_cost)
        print(" ")
        validation_time.append(time_cost)
    validation_time_total.append(validation_time)
    k_acc_total.append(k_acc)

# draw each validation fig
for k in range(len(metric_list)):
    draw_figure(range(1, k_range_toplimit), k_acc_total[k], "k for KNN", "accuracy", "dataset proportion: "
                + data_set_proportion.__str__() + " & metric: " + metric_list[k] + " & k for KNN ACCURACY")
    draw_figure(range(1, k_range_toplimit), validation_time_total[k], "k for KNN", "time cost", "dataset proportion: "
                + data_set_proportion.__str__() + " & metric: " + metric_list[k] + " & k for KNN TIME COST")

# draw the comperation fig of all validation accuracy
color = ["red", "green", "blue", "yellow"]
plt.title("dataset proportion: " + data_set_proportion.__str__() + " ACCURACY")
plt.grid(True, linestyle='--')
plt.xlabel("k for KNN")
plt.ylabel("accuracy")
my_x_ticks = np.arange(0, k_range_toplimit, 1)
plt.xticks(my_x_ticks)
for k in range(len(metric_list)):
    plt.plot(range(1, k_range_toplimit), k_acc_total[k], label=metric_list[k], marker="",
             linestyle="-", color=color[k])
plt.legend(prop={'size': 10})
plt.savefig(os.path.split(os.path.realpath(__file__))[0] + "/plt/" + "Validation Comparation & dataset proportion: "
            + data_set_proportion.__str__() + ' ACCURACY.png', bbox_inches='tight')
plt.show()

# draw the comperation fig of all validation time cost
color = ["red", "green", "blue", "yellow"]
plt.title("dataset proportion: " + data_set_proportion.__str__() + " TIME COST")
plt.grid(True, linestyle='--')
plt.xlabel("k for KNN")
plt.ylabel("time cost")
my_x_ticks = np.arange(0, k_range_toplimit, 1)
plt.xticks(my_x_ticks)
for k in range(len(metric_list)):
    plt.plot(range(1, k_range_toplimit), validation_time_total[k], label=metric_list[k], marker="",
             linestyle="-", color=color[k])
plt.legend(prop={'size': 10})
plt.savefig(os.path.split(os.path.realpath(__file__))[0] + "/plt/" + "Validation Comparation & dataset proportion: "
            + data_set_proportion.__str__() + ' TIME COST.png', bbox_inches='tight')
plt.show()