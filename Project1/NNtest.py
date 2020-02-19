import NearestNeighbor as NN
from DataExtractor import *
import numpy as np

batch_list, batch_test = get_batch_data()

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
