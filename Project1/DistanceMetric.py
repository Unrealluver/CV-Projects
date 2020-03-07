import numpy as np
from math import *

def euclidean(x, y):
    distances = sqrt(np.sum((y - x)**2, axis=1))
    return distances

def manhattan(x, y):
    distances = np.sum(np.abs(y - x), axis=1)
    return distances

def chebyshev(x, y):
    distances = np.max(np.abs(y - x), axis=1)
    return distances

def cosine(x, y):
    distances = 1 - np.dot(y, x) / (np.linalg.norm(y, axis=1) * np.linalg.norm(x))
    return distances