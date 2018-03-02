# -*- coding: utf-8 -*-

import abc
import tensorflow as tf


class BaseCNN(metaclass=abc.ABCMeta):

    def __init__(self, vocab_size, embeddings_size, filter_sizes,
                 num_filters, num_classes, lr_method='adam', sequence_length=None):
        self.add_placeholders(sequence_length)
        self.add_embeddings_layer(vocab_size, embeddings_size)
        self.add_cnn_layer(filter_sizes, embeddings_size, num_filters, sequence_length)
        self.add_regularization()
        self.add_dense_layer(num_classes)
        self.add_loss()
        self.add_train_op(lr_method)
        self.initialize_session()

    @abc.abstractmethod
    def add_embeddings_layer(self, vocab_size, embeddings_size):
        return

    @abc.abstractmethod
    def add_regularization(self):
        return

    @abc.abstractmethod
    def add_dense_layer(self, num_classes):
        return

    def add_placeholders(self, sequence_length=None):
        if sequence_length is None:  # padding on the fly
            self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")
        else:  # padding done before
            self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def add_cnn_layer(self, filter_sizes, embedding_size, num_filters, sequence_length):
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                # Convolution Layer
                filter_shape = [filter_size,
                                embedding_size,
                                self.num_channels,
                                num_filters]
                # filter_shape = [filter_size, embedding_size, 2, num_filters]  # 2 if we use 2 channels ??
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_words_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply non-linearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-pooling over the outputs
                if sequence_length is None:  # padding on the fly, don't know yet if it can work...
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, None, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                else:
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        self.num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

    def add_loss(self):
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

    def add_train_op(self, lr_method):
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")
        _lr_m = lr_method.lower()  # lower to make sure
        with tf.variable_scope("train_step"):
            if _lr_m == 'adam':  # sgd method
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))
            self.train_op = optimizer.minimize(self.loss)

    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        print("Initializing tf session")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
