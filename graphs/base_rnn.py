# -*- coding: utf-8 -*-

import abc
import tensorflow as tf


class BaseRNN(metaclass=abc.ABCMeta):

    def __init__(self, vocab_size, embeddings_size, num_classes, lr_method='adam', sequence_length=None):
        self.add_placeholders()
        self.add_embeddings_layer()
        self.add_rnn_layer()

    @abc.abstractmethod
    def add_embeddings_layer(self):
        return


