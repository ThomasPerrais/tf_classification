# -*- coding: utf-8 -*-

import tensorflow as tf


class TextClassificationCNN(object):

    def __init__(self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, lr_method='adam'):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")  # Maybe use [None, None] and pad on the fly ?
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.populate_embeddings_layer(vocab_size, embedding_size)

        self.populate_cnn(embedding_size, sequence_length, num_filters, filter_sizes)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        self.populate_dense_layers(num_classes)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        self.add_train_op(lr_method)
        self.initialize_session()

    def populate_embeddings_layer(self, vocab_size, embedding_size):

        # with tf.device('/cpu:0'), tf.name_scope("embedding"):
        with tf.name_scope("embedding"):

            # w_to_train = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            #                          name="W_to_train")
            # w_trained = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size],
            #                         trainable=False, name="W_trained"))
            #
            # self.embedding_trained_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
            # self.embedding_init = w_trained.assign(self.embedding_trained_placeholder)
            #
            # self.embedded_words_to_train = tf.nn.embedding_lookup(w_to_train, self.input_x)
            # self.embedded_words_trained = tf.nn.embedding_lookup(w_trained, self.input_x)
            #
            # self.embedded_words_expanded = tf.concat(3, [self.embedded_words_to_train, self.embedded_words_trained])

            # ONLY PRE-TRAINED VECTORS :

            w_trained = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]),
                                    trainable=False, name="w_trained")

            self.embedding_trained_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
            self.embedding_init = w_trained.assign(self.embedding_trained_placeholder)

            self.embedded_words_trained = tf.nn.embedding_lookup(w_trained, self.input_x)

            self.embedded_words_expanded = tf.expand_dims(self.embedded_words_trained, -1)


    def populate_cnn(self, embedding_size, sequence_length, num_filters, filter_sizes):

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # filter_shape = [filter_size, embedding_size, 2, num_filters]  # 2 if we use 2 chanels ??
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


    def populate_dense_layers(self, num_classes):

        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([self.num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")


    def initialize_session(self):

        """Defines self.sess and initialize the variables"""
        print("Initializing tf session")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


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
