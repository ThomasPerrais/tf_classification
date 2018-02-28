# -*- coding: utf-8 -*-

import codecs
import os
import sys

import numpy as np

from model.cnn_classification import TextClassificationCNN
from preprocessing.preprocessor import load_vocabulary
from utils.progress import Progbar


class TextClassifier(object):

    def __init__(self, directory, filter_sizes, num_filters=100, dropout_keep_prob=0.5, batch_size=50, init_lr=0.025):

        self.directory = directory
        self.dropout_keep_prob = dropout_keep_prob
        self.batch_size = batch_size
        self.best_accuracy = 0
        self.init_lr = init_lr

        # Loading vocabulary as dict { word : id }
        self.vocabulary = load_vocabulary(os.path.join(self.directory, "vocabulary.txt"))

        # Loading training set
        self.x_train, self.y_train, self.x_dev, self.y_dev = load_training_set(
            os.path.join(self.directory, "training.int.txt"))
        self.train_size = self.x_train.shape[0]
        self.dev_size = self.x_dev.shape[0]

        # Loading embeddings
        self.embeddings = np.load(os.path.join(self.directory, "embeddings.npy"))

        # Creating model
        self.model = TextClassificationCNN(
            sequence_length=self.x_train.shape[1],
            num_classes=np.max(self.y_train) + 1,
            vocab_size=self.embeddings.shape[0],
            embedding_size=self.embeddings.shape[1],
            filter_sizes=filter_sizes,
            num_filters=num_filters)

    def train_model(self, max_epochs=100):
        for e in range(max_epochs):
            new_best = self.run_epoch(e, max_epochs)
            if new_best:
                self.save_current_model()

    def save_current_model(self):
        # TODO : save the weights of the model
        return

    def evaluate_on_dev(self):
        accuracies = []
        for i, (x_batch, y_batch) in enumerate(self.minibatches(train=False)):
            _, dev_accuracy = self.process_batch(x_batch, y_batch, self.init_lr, dropout=1.0, need_accuracy=True)
            accuracies.append(dev_accuracy)
        mean_acc = np.mean(accuracies)
        print("\nscore on dev set : {}".format(mean_acc))
        if mean_acc > self.best_accuracy:
            print(' - new best score !')
            return True
        else:
            return False

    def process_batch(self, x_batch, y_batch, lr, dropout=-1., need_accuracy=False):
        if dropout < 0:
            dropout = self.dropout_keep_prob
        feed_dict = {
            self.model.input_x: x_batch,
            self.model.input_y: y_batch,
            self.model.dropout_keep_prob: dropout,
            self.model.embedding_trained_placeholder: self.embeddings,
            self.model.lr: lr
        }
        _, loss, scores = self.model.sess.run(
            [self.model.train_op, self.model.loss, self.model.scores],
            feed_dict)
        accuracy = 0
        if need_accuracy:
            accuracy = sum([np.argmax(score) == y for score, y in zip(scores, y_batch)])
        return loss, accuracy

    def run_epoch(self, epoch, max_epochs):
        """Performs one complete pass over the train set and evaluate on dev"""
        nbatches = self.train_size // self.batch_size
        prog = Progbar(target=nbatches)

        lr = self.init_lr * (1 - float(epoch)/max_epochs)
        # iterate over dataset
        for i, (x_batch, y_batch) in enumerate(self.minibatches(train=True)):
            train_loss, _ = self.process_batch(x_batch, y_batch, lr)
            prog.update(i + 1, [("train loss", train_loss)])
            # print_progress_and_infos(epoch, i + 1, nbatches, train_loss)
        return self.evaluate_on_dev()

    def minibatches(self, train):
        if train:
            m = np.concatenate((self.x_train, self.y_train.reshape(-1, 1)), axis=1)
        else:
            m = np.concatenate((self.x_dev, self.y_dev.reshape(-1, 1)), axis=1)
        np.random.shuffle(m)
        nbatches = m.shape[0] // self.batch_size
        for b in range(nbatches):
            yield m[b * self.batch_size:b * self.batch_size + self.batch_size, :-1],\
                  m[b * self.batch_size:b * self.batch_size + self.batch_size, -1]


def load_training_set(filename, prop=0.1):
    """
    Create the numpy array containing the training sentences and associated training classes,
    separate in training and dev sets
    :param filename: path to the training set file
    :param prop: proportion of training sample to keep for calculating dev score
    :return: Four numpy array
     x_train and x_dev where each line is a padded sentence,
     y_train and y_dev containing classes assignment
    """
    with codecs.open(filename, "r", encoding='utf-8') as f:
        i = -1
        for line in f.read().splitlines():
            if i == -1:
                sizes = [int(elt) for elt in line.strip().split('\t')]
                if len(sizes) != 2:
                    print("model does not support {}-d arrays".format(len(sizes)))
                x_train = np.zeros((sizes[0], sizes[1]))
            else:
                x_train[i, :] = [int(elt) for elt in line.strip().split(' ')]
            i += 1
    x_train = x_train.astype(int)
    # Randomly shuffle data
    np.random.seed(10)
    np.random.shuffle(x_train)
    threshold = int(x_train.shape[0] * (1 - prop))
    return x_train[:threshold, :-1], x_train[:threshold, -1], x_train[threshold:, -1], x_train[threshold:, -1]


def print_progress_and_infos(epoch, i, nbatches, loss, size=60):

    percent = i * size // nbatches
    print('Epoch {} [{}>{}] - train loss : {:0.4f}'.format(epoch, '=' * percent, '-' * (size - percent), loss))


def main(args):

    filters = [3, 4, 5]
    epochs = 100
    if len(args) == 0:
        print("need to specify a folder where training set is stored")
        sys.exit(2)
    directory = os.path.expanduser('~/Documents/Projects/Classification/{}'.format(args[0]))
    if len(args) > 1:
        extension = args[1]
        directory = os.path.join(directory, extension)
    classifier = TextClassifier(directory, filters)
    if len(args) > 2:
        epochs = int(args[2])
    classifier.train_model(epochs)

if __name__ == "__main__":
    main(sys.argv[1:])
