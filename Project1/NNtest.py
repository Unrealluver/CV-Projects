import NearestNeighbor as NN
import numpy as np

batch_list = []
data_directory = "./cifar-10-python/cifar-10-batches-py/"

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

for i in range(1, 6):
    batch_list.append(unpickle(data_directory + "data_batch_" + i.__str__()))

batch_test = unpickle(data_directory + "test_batch")

nn_model = NN.NearestNeighbor()

for batch in batch_list:
    nn_model.train(batch.get(b'data'), batch.get(b'labels'))
    print(str(batch.get(b'batch_label')) + ' has been trained done!')

print('predicting...')
data_set_proportion = 0.1
prediction = nn_model.predict(batch_test.get(b'data'), data_set_proportion)
print('prediction has bend done.')

result = prediction - batch_test.get(b'labels')[:int(data_set_proportion * len(batch_test.get(b'labels')))]

true = 0

for i in range(len(result)):
    if result[i] == 0:
        true +=1

print('final acc of Nearest Neighbor classifier is: ' + '%.2f%%'%(100*(true / len(result))))
