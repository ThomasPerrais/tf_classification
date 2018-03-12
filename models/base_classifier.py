# -*- coding: utf-8 -*-

import abc

# This file is meant to define the function that a text classifier should implement


class AbstractTextClassifier(metaclass=abc.ABCMeta):

    def __init__(self, directory, batch_size, dropout_keep_prob, init_lr, verbose):
        self.verbose = verbose
        self.directory = directory
        self.batch_size = batch_size
        self.dropout_keep_prob = dropout_keep_prob
        self.init_lr = init_lr

    @abc.abstractmethod
    def train_model(self, max_epochs):
        return

    @abc.abstractmethod
    def save_model(self):
        return

    @abc.abstractmethod
    def evaluate_on_dev(self):
        return

    @abc.abstractmethod
    def predict_batch(self, x_batch, y_batch=None, lr=None, dropout=-1.):
        return

    @abc.abstractmethod
    def run_epoch(self, current_epoch, max_epochs):
        return

    @abc.abstractmethod
    def restore_model(self):
        return

    @abc.abstractmethod
    def confusion_matrix(self, trainset):
        return

    @abc.abstractmethod
    def evaluate(self, testset, soft=True):
        return
