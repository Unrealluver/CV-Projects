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

