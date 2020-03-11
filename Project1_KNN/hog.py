from typing import List, Any

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
import numpy as np
from DataExtractor import *
from tqdm import tqdm


def get_hog_data(train_X, train_y, test_X, test_y, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(4, 4), multichannel=True):
    train_X = np.reshape(train_X, (-1, 3, 32, 32))
    train_X = np.transpose(train_X, (0, 2, 3, 1))
    train_X_hog: List[Any] = []
    for i in tqdm(range(train_X.shape[0])):
        train_X_hog.append(
            hog(train_X[i], orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, multichannel=multichannel))
    train_X_hog = np.array(train_X_hog, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.int32)

    test_X = np.reshape(test_X, (-1, 3, 32, 32))
    test_X = np.transpose(test_X, (0, 2, 3, 1))
    test_X_hog = []
    for i in tqdm(range(test_X.shape[0])):
        test_X_hog.append(
            hog(test_X[i], orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, multichannel=multichannel))
    test_X_hog = np.array(test_X_hog, dtype=np.float32)
    test_y = np.array(test_y, dtype=np.int32)

    print(np.shape(train_X_hog), ' /// ', np.shape(train_y), ' /// ', np.shape(test_X_hog), ' /// ', np.shape(test_y))
    return train_X_hog, train_y, test_X_hog, test_y
