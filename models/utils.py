# -*- coding: utf-8 -*-

from sklearn.preprocessing import LabelBinarizer
import numpy as np


def load_training_set(filename, prop=0.1, flat=True):
    """
    Create the numpy array containing the training sentences and associated training classes,
    separate in training and dev sets
    :param filename: path to the training set file
    :param prop: proportion of training sample to keep for calculating dev score
    :param flat: to chose between 1d and 2d array of 0 and 1 for y.
    :return: Four numpy array
     x_train and x_dev where each line is a padded sentence,
     y_train and y_dev containing classes assignment
    """
    x_train = np.load(filename)
    x_train = x_train.astype(int)

    # Randomly shuffle data
    np.random.seed(10)
    np.random.shuffle(x_train)
    threshold = int(x_train.shape[0] * (1 - prop))

    if flat:
        y_train = x_train[:, -1]
    else:
        lb = LabelBinarizer()
        y_train = lb.fit_transform(x_train[:, -1])

    return x_train[:threshold, :-1], y_train[:threshold], x_train[threshold:, :-1], y_train[threshold:]
