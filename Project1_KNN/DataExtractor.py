import numpy as np
import os
import platform
import matplotlib.pyplot as plt

batch_list = []
data_directory = "../cifar-10-batches-py/"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_pickle(f):
    from six.moves import cPickle as pickle
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='bytes')
    raise ValueError("invalid python version: {}".format(version))

'''
data files contain three binary part.
batch_label: which represents the index of the dataset.
labels: which represents the category of the images, 
        the index of the row is also the row index of the image.
data: which is the image data that has 3072 columns, 
        the first 1024 columns represent the red channels of the 32x32 image, 
        the next 1024 columns represent the green one, 
        the last represent the blue one.
filenames: which is the name of the images.
'''

# batch_list = unpickle(data_directory + "data_batch_1")
def get_batch_data():
    for i in range(1, 6):
        batch_list.append(unpickle(data_directory + "data_batch_" + i.__str__()))

    batch_test = unpickle(data_directory + "test_batch")
    return batch_list, batch_test

def get_all_data():
    batch_list, batch_test = get_batch_data()
    # train_X = [batch.get(b'data') for batch in batch_list]
    # train_y = [batch.get(b'labels') for batch in batch_list]
    train_X = np.array(batch_list[0].get(b'data'))
    train_y = np.array(batch_list[0].get(b'labels'))

    for i in range(1, len(batch_list)):
        train_X = np.concatenate((train_X, batch_list[i].get(b'data')), axis=0)
        train_y = np.concatenate((train_y, batch_list[i].get(b'labels')), axis=0)

    test_X = batch_test.get(b'data')
    test_y = batch_test.get(b'labels')
    return train_X, train_y, test_X, test_y


def get_normalized_data( num_training=49000, num_validation=1000, num_test=100, num_dev=100):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier.
    """
    X_train, y_train, X_test, y_test = get_all_data()

    # subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[num_training: num_training + num_validation]
    y_val = y_train[num_training: num_training + num_validation]
    mask = list(range(num_training))
    X_train = X_train[0: num_training]
    y_train = y_train[0: num_training]
    mask = list(range(num_test))
    X_test = X_test[0: num_test]
    y_test = y_test[0: num_test]
    masks = np.random.choice(num_training, num_dev, replace=False)
    X_dev = []
    y_dev = []
    for mask in masks:
        X_dev = np.hstack((X_dev, X_train[mask]))
        y_dev = np.hstack((y_dev, y_train[mask]))

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    # X_train = np.array(X_train, dtype=float)
    # X_val = np.array(X_val, dtype=float)
    # X_test = np.array(X_test, dtype=float)
    # X_dev = np.array(X_dev, dtype=float)

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    # mean_image = mean_image.reshape(1, -1)
    # print(mean_image.shape, " / ", X_train.shape, " / ", X_train[1, :].shape)
    # for i in range(X_train.shape[0]):
    #     X_train[i, :] -= mean_image.reshape(1, 3072)
    # for i in range(X_val.shape[0]):
    #     X_val[:, i] -= mean_image
    # for i in range(X_test.shape[0]):
    #     X_test[:, i] -= mean_image
    # for i in range(X_dev.shape[0]):
    #     X_dev[:, i] -= mean_image

    # 均一化
    X_train = X_train.tolist()
    X_val = X_val.tolist()
    X_test = X_test.tolist()
    X_dev = X_dev.tolist()

    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    X_dev = np.array(X_dev)

    X_train /= 128
    X_val /= 128
    X_test /= 128
    X_dev /= 128

    # # add bias dimension and transform into columns
    # X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    # X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    # X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    # X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev

def get_one_hot_label(labels, label_num):
    one_hot_labels = []
    for label in labels:
        token = np.zeros(label_num)
        token[label - 1] = 1
        one_hot_labels.append(token)
    return one_hot_labels


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict[b'data']
        # print("X's shape: ", np.shape(X))
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        print("xs' shape: ", np.shape(xs))
        ys.append(Y)
    Xtr = np.concatenate(xs)
    print("Xtr' shape: ", np.shape(Xtr))
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(path_to_cifar_dir='../cifar-10-batches-py',
                     num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    X_train, y_train, X_test, y_test = load_CIFAR10(path_to_cifar_dir)
    print("X_train's shape(just load)", np.shape(X_train))

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # fig = plt.figure()
    # plt.imshow(X_train[0])
    # plt.show()
    # print("X_train[0]'s shape", np.shape(X_train[0]))
    # print(y_train[0])
    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # fig = plt.figure()
    # plt.imshow(X_train[0])
    # plt.show()
    # print("X_train[0]'s shape", np.shape(X_train[0]))
    # print(y_train[0])

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    # print("X_train's shape", np.shape(X_train))
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()
    #
    # Package data into a dictionary
    # return {
    #   'X_train': X_train, 'y_train': y_train,
    #   'X_val': X_val, 'y_val': y_val,
    #   'X_test': X_test, 'y_test': y_test,
    # }

    return X_train, y_train, X_val, y_val, X_test, y_test