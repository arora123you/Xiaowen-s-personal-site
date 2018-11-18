import numpy as np
from scipy.misc import imread, imresize
from sklearn import preprocessing

def preprocess_input(x):
    x = x.astype('float32')
    x.resize(48,144)
    x -= np.mean(x, axis = 0)
    x /= np.std(x, axis = 0)
    x.resize(48,48,3)
    return x

def _imread(image_name):
        return imread(image_name)

def _imresize(image_array, size):
        return imresize(image_array, size)

def to_categorical(integer_classes, num_classes=2):
    integer_classes = np.asarray(integer_classes, dtype='int')
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes))
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical

