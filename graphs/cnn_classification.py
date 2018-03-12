# -*- coding: utf-8 -*-

import tensorflow as tf
from graphs.base_cnn import BaseCNN


class MultiChannelCNN(BaseCNN):

    def __init__(self, vocab_size, embeddings_size, filter_sizes, num_filters,
                 num_classes, lr_method='adam', sequence_length=None):
        self.num_channels = 2
        super(MultiChannelCNN, self).__init__(vocab_size, embeddings_size, filter_sizes,
                                num_filters, num_classes, lr_method, sequence_length)

    def add_embeddings_layer(self, vocab_size, embeddings_size):
        # with tf.device('/cpu:0'), tf.name_scope("embedding"):
        with tf.name_scope("embedding"):
            w_to_train = tf.Variable(tf.random_uniform([vocab_size, embeddings_size], -1.0, 1.0),
                                     name="W_to_train")
            w_trained = tf.Variable(tf.constant(0.0, shape=[vocab_size, embeddings_size]),
                                    trainable=False, name="W_trained")

            self.embedding_trained_placeholder = tf.placeholder(tf.float32, [vocab_size, embeddings_size])
            self.embedding_init = w_trained.assign(self.embedding_trained_placeholder)

            self.embedded_words_to_train = tf.nn.embedding_lookup(w_to_train, self.input_x)
            self.embedded_words_trained = tf.nn.embedding_lookup(w_trained, self.input_x)

            self.embedded_words_to_train_expanded = tf.expand_dims(self.embedded_words_to_train, -1)
            self.embedded_words_trained_expanded = tf.expand_dims(self.embedded_words_trained, -1)

            self.embedded_words_expanded = tf.concat([self.embedded_words_to_train_expanded,
                                                      self.embedded_words_trained_expanded], 3)

    def add_regularization(self):
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

    def add_dense_layer(self, num_classes):
        """Create a one layer dense neural network to make prediction"""
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([self.num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")


class FixedVectorsCNN(BaseCNN):

    def __init__(self, vocab_size, embeddings_size, filter_sizes, num_filters,
                 num_classes, lr_method='adam', sequence_length=None):
        self.num_channels = 1
        super(FixedVectorsCNN, self).__init__(vocab_size, embeddings_size, filter_sizes,
                                num_filters, num_classes, lr_method, sequence_length)

    def add_embeddings_layer(self, vocab_size, embeddings_size):
        """ Using pre-trained word embeddings as input of the CNN. """
        w_trained = tf.Variable(tf.constant(0.0, shape=[vocab_size, embeddings_size]),
                                trainable=False, name="w_trained")
        self.embedding_trained_placeholder = tf.placeholder(tf.float32, [vocab_size, embeddings_size])
        self.embedding_init = w_trained.assign(self.embedding_trained_placeholder)
        self.embedded_words_trained = tf.nn.embedding_lookup(w_trained, self.input_x)
        self.embedded_words_expanded = tf.expand_dims(self.embedded_words_trained, -1)

    def add_regularization(self):
        """ Adding dropout on the concatenation of vectors from the CNN. """
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

    def add_dense_layer(self, num_classes):
        """Adding a one layer dense neural network to make prediction"""
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([self.num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")


class TrainableVectorsCNN(BaseCNN):

    def __init__(self, vocab_size, embeddings_size, filter_sizes, num_filters,
                 num_classes, lr_method='adam', sequence_length=None):
        self.num_channels = 1
        super(TrainableVectorsCNN, self).__init__(vocab_size, embeddings_size, filter_sizes,
                                num_filters, num_classes, lr_method, sequence_length)

    def add_regularization(self):
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

    def add_dense_layer(self, num_classes):
        """Create a one layer dense neural network to make prediction"""
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([self.num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

    def add_embeddings_layer(self, vocab_size, embeddings_size):
        # with tf.device('/cpu:0'), tf.name_scope("embedding"):
        with tf.name_scope("embedding"):
            w_to_train = tf.Variable(tf.random_uniform([vocab_size, embeddings_size], -1.0, 1.0),
                                     name="W_to_train")
            self.embedded_words = tf.nn.embedding_lookup(w_to_train, self.input_x)
            self.embedded_words_expanded = tf.expand_dims(self.embedded_words, -1)
