# -*- coding: utf-8 -*-

import os
import numpy as np
import math
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from build_data import EMBEDDINGS_OUTPUT_NAME, VOCABULARY_NAME, TRAIN_OUTPUT_NAME, CLASSES_NAME

from graphs.cnn_classification import FixedVectorsCNN, TrainableVectorsCNN, MultiChannelCNN
from preprocessing.preprocessor import load_vocabulary, load_classes
from utils.progress import Progbar
from .utils import load_training_set
from .base_classifier import AbstractTextClassifier


class TextClassifier(AbstractTextClassifier):

    def __init__(self, directory, filter_sizes, num_filters,
                 dropout_keep_prob, batch_size, init_lr, verbose, mode="trainable"):
        super(TextClassifier, self).__init__(directory, batch_size, dropout_keep_prob, init_lr, verbose)
        self.mode = mode
        self.best_accuracy = 0

        # Loading vocabulary as dict { word : id }
        self.vocabulary = load_vocabulary(os.path.join(self.directory, VOCABULARY_NAME))

        # Loading training set
        self.x_train, self.y_train, self.x_dev, self.y_dev = load_training_set(
            os.path.join(self.directory, TRAIN_OUTPUT_NAME))
        self.train_size = self.x_train.shape[0]
        self.dev_size = self.x_dev.shape[0]

        self.num_classes = max(np.max(self.y_train), np.max(self.y_dev)) + 1

        # Loading embeddings
        self.embeddings = np.load(os.path.join(self.directory, EMBEDDINGS_OUTPUT_NAME))

        # Creating graphs
        if self.mode == "trainable":
            self.model = TrainableVectorsCNN(
                sequence_length=self.x_train.shape[1],
                num_classes=self.num_classes,
                vocab_size=self.embeddings.shape[0],
                embeddings_size=self.embeddings.shape[1],
                filter_sizes=filter_sizes,
                num_filters=num_filters
            )
        elif self.mode == "fixed":
            self.model = FixedVectorsCNN(
                sequence_length=self.x_train.shape[1],
                num_classes=np.max(self.y_train) + 1,
                vocab_size=self.embeddings.shape[0],
                embeddings_size=self.embeddings.shape[1],
                filter_sizes=filter_sizes,
                num_filters=num_filters
            )
        elif self.mode == "both":
            self.model = MultiChannelCNN(
                sequence_length=self.x_train.shape[1],
                num_classes=np.max(self.y_train) + 1,
                vocab_size=self.embeddings.shape[0],
                embeddings_size=self.embeddings.shape[1],
                filter_sizes=filter_sizes,
                num_filters=num_filters
            )
        else:
            raise Exception("mode {} is not supported".format(self.mode))

        self.save_dir = os.path.join(self.directory,
                                     "weights", "{}_f{}_nf{}".format(self.mode,
                                                                     "-".join([str(elt) for elt in filter_sizes]),
                                                                     str(num_filters)))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def confusion_matrix(self, trainset):
        y_pred = []
        y_true = []
        for x_batch in self.minibatches(x=trainset):
            y = x_batch[:, -1]
            x = x_batch[:, :-1]
            _, pred = self.predict_batch(x)
            y_true.extend(y)
            y_pred.extend(pred)
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        return cm

    def save_confusion_matrix(self, cm):
        path_classes = os.path.join(self.directory, CLASSES_NAME)
        classes = load_classes(path_classes)
        if cm.shape[0] != len(classes):
            print('incompatibility between classes files ({} classes) '
                  'and confusion matrix ({} classes)'.format(len(classes), cm.shape[0]))
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        plt.figure(figsize=(13, 8))
        ax = plt.axes()
        sn_heatmap = sn.heatmap(df_cm, annot=True, ax=ax)
        ax.set_xlabel("predicted label")
        ax.set_ylabel("true label")
        figure = sn_heatmap.get_figure()
        figure.savefig(os.path.join(self.save_dir, "confusion_matrix.png"))

    def train_model(self, max_epochs=100):
        if not self.mode == "trainable":
            self.model.sess.run(self.model.embedding_init,
                                feed_dict={self.model.embedding_trained_placeholder: self.embeddings})
        for e in range(max_epochs):
            new_best = self.run_epoch(e, max_epochs)
            if new_best:
                cm = self.confusion_matrix(np.concatenate((self.x_dev, self.y_dev.reshape(-1, 1)), axis=1))
                self.save_confusion_matrix(cm)
                self.save_model()

    def save_model(self):
        """Saves session = weights"""
        self.model.saver.save(self.model.sess, os.path.join(self.save_dir, "model"))

    def evaluate_on_dev(self):
        good_preds = 0
        for x_batch, y_batch in self.minibatches(train=False):
            _, pred = self.predict_batch(x_batch, dropout=1.)
            good_preds += sum([p == t for p, t in zip(pred, y_batch)])
        mean_acc = 100. * good_preds / self.x_dev.shape[0]
        print("\nscore on dev set : {}".format(mean_acc))
        if mean_acc > self.best_accuracy:
            self.best_accuracy = mean_acc
            print(' - new best score !')
            return True
        else:
            return False

    def get_params(self, x_batch, y_batch=None, lr=None, dropout=-1.):
        if dropout < 0:
            dropout = self.dropout_keep_prob
        feed_dict = {
            self.model.input_x: x_batch,
            self.model.dropout_keep_prob: dropout,
            self.model.lr: lr,
        }
        if y_batch is not None:
            feed_dict[self.model.input_y] = y_batch
        return feed_dict

    def predict_batch(self, x_batch, y_batch=None, lr=None, dropout=-1.):
        feed_dict = self.get_params(x_batch, y_batch, lr, dropout)

        scores, pred = self.model.sess.run(
            [self.model.scores, self.model.predictions],
            feed_dict
        )
        return scores, pred

    def run_epoch(self, epoch, max_epochs):
        """Performs one complete pass over the train set and evaluate on dev"""
        nbatches = math.ceil(self.train_size / self.batch_size)
        prog = Progbar(target=nbatches)

        lr = self.init_lr * (1 - float(epoch)/max_epochs)
        # iterate over dataset
        for i, (x_batch, y_batch) in enumerate(self.minibatches(train=True)):
            feed_dict = self.get_params(x_batch, y_batch, lr)

            _, train_loss = self.model.sess.run(
                [self.model.train_op, self.model.loss],
                feed_dict)

            if self.verbose:
                prog.update(i + 1, [("train loss", train_loss)])
        return self.evaluate_on_dev()

    def minibatches(self, train=True, x=None, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if x is not None:
            nbatches = x.shape[0] // self.batch_size
            for b in range(nbatches):
                yield x[b * batch_size:b * batch_size + batch_size, :]
            if nbatches * batch_size < x.shape[0]:
                yield x[nbatches * batch_size:, :]
        else:
            if train:
                m = np.concatenate((self.x_train, self.y_train.reshape(-1, 1)), axis=1)
            else:
                m = np.concatenate((self.x_dev, self.y_dev.reshape(-1, 1)), axis=1)
            np.random.shuffle(m)
            nbatches = m.shape[0] // batch_size
            for b in range(nbatches):
                yield m[b * batch_size:b * batch_size + batch_size, :-1],\
                      m[b * batch_size:b * batch_size + batch_size, -1]
            if nbatches * batch_size < m.shape[0]:
                yield m[nbatches * batch_size:, :-1], \
                      m[nbatches * batch_size:, -1]

    def restore_model(self):
        if self.verbose:
            print("Reloading the latest trained model...")
        self.model.saver.restore(self.model.sess, os.path.join(self.save_dir, "model"))

    def evaluate(self, testset, soft=True):
        """
        evaluate current model on the testset
        :param testset: numpy array where each line is a sentence
        :param soft: true to return the classes probabilities, false to return classes assignment.
        :return: numpy array where each line is the prediction for the corresponding sentence
        """
        current = 0
        if soft:
            results = np.zeros((testset.shape[0], self.num_classes))
        else:
            results = np.zeros((testset.shape[0], 1))
        for x_batch in self.minibatches(x=testset):
            if soft:
                score, _ = self.predict_batch(x_batch, dropout=1.)
                results[current:current + x_batch.shape[0], :] = score
            else:
                _, score = self.predict_batch(x_batch, dropout=1.)
                results[current:current + x_batch.shape[0], :] = score.reshape(-1, 1)
            current += x_batch.shape[0]
        return results
