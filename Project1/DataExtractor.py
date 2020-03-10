import numpy as np

batch_list = []
data_directory = "../cifar-10-batches-py/"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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


def get_linear_data( num_training=49000, num_validation=1000, num_test=100, num_dev=100):
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

    # add bias dimension and transform into columns
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev