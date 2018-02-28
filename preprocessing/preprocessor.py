# -*- coding: utf-8 -*-

import codecs
import gensim
import numpy as np

PAD = "<P_A_D>"


class Preprocessor(object):

    def __init__(self, path, train=True, processing_words=None, processing_class=None,
                 max_iter=None, max_length=-1):
        self.path = path
        self.train = train
        self.processing_words = processing_words
        self.processing_class = processing_class
        self.max_iter = max_iter
        self.length = None

    def __iter__(self):
        n_iter = 0
        with codecs.open(self.path, "r", encoding='utf-8') as f:
            for line in f:
                splitted = line.strip().split(' ')
                id = splitted[0]
                if self.train:
                    topic = splitted[1]
                    words = [elt for elt in splitted[2:] if len(elt) > 0]
                else:
                    topic = None
                    words = [elt for elt in splitted[1:] if len(elt) > 0]
                if self.processing_words is not None:
                    words = self.processing_words(words)
                if self.max_iter is not None and n_iter > self.max_iter:
                    break
                if self.processing_class is not None and topic is not None:
                    topic = self.processing_class(topic)
                n_iter += 1
                yield id, topic, words

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_vocabulary(preprocessors):
    """
    return the vocabulary associated with a given list of files containing sentences
    as well as the size of the longest sentence
    :param preprocessors: list of preprocessors to extract vocab from
    :return: two sets : vocabulary and list of classes
    """
    print("Creating vocabulary... ")
    vocab_words = set()
    vocab_classes = set()
    max_length = 0
    for preprocessor in preprocessors:
        for _, topic, words in preprocessor:
            if topic is not None:
                vocab_classes.add(topic)
            if len(words) > max_length:
                max_length = len(words)
            vocab_words.update(words)
    print("- done. {} tokens - {} classes".format(len(vocab_words), len(vocab_classes)))
    return vocab_words, vocab_classes, max_length


def write_vocabulary(vocab, filename):
    """
    Save the given vocabulary, one word per line.
    :param vocab: list of words
    :param filename: path to save the vocabulary
    :return: nothing
    """
    print("Writing vocab... ")
    with codecs.open(filename, "w", encoding='utf-8') as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write(word)
                f.write("\n")
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocabulary(filename):
    """
    Read a vocabulary file with one word per line and return the corresponding dictionary
    :param filename: path where vocabulary is stored
    :return: dictionary {word : id}
    """
    with codecs.open(filename, "r", encoding='utf-8') as f:
        d = {elt.strip(): i for i, elt in enumerate(f)}
    return d


def create_embeddings(vocab, embeddings_size, filename_w2v, filename_ngrams_w2v, filename_save,
                      min_n, max_n, start_char="<", end_char=">"):
    """
    Create the matrix associated with the given vocab. if the word is present in the embeddings vocabulary
    the associated vector is chosen, otherwise vector is derived using character ngrams
    :param vocab: dictionary {word, id}
    :param embeddings_size : len of initial vocabulary (might be bigger than len(vocab)...)
    :param filename_w2v: path to word embeddings
    :param filename_ngrams_w2v: path to char ngrams embeddings
    :param filename_save: path to save the matrix that will be created
    :param min_n: min size of char ngrams
    :param max_n: max size of char ngrams
    :param start_char: char that define a beginning of word in the embeddings of chars ngrams
    :param end_char: char that define an end of word in the embeddings of chars ngrams
    :return: nothing
    """
    print("Creating embeddings using found vocab of length : {}".format(len(vocab)))

    # Load embeddings
    print("\nLoading embeddings ... ")
    w2v = gensim.models.KeyedVectors.load_word2vec_format(filename_w2v, binary=True, unicode_errors='ignore')
    w2v_ngrams = gensim.models.KeyedVectors.load_word2vec_format(filename_ngrams_w2v, binary=True, unicode_errors='ignore')

    embedded_vocab = set(w2v.vocab.keys())
    embedded_ngrams = set(w2v_ngrams.vocab.keys())

    print(" - done. vocabulary size : {}; ngrams size : {}".format(len(embedded_vocab), len(embedded_ngrams)))

    count_present = 0
    count_absent = 0
    count_absent_small = 0

    vectors = np.zeros((embeddings_size, w2v.vector_size))
    for word in vocab:
        index = vocab[word]
        if word in embedded_vocab:
            count_present += 1
            vectors[index, :] = w2v[word]
        else:
            ngrams = get_char_ngrams(word, min_n, max_n, start_char, end_char)
            count_ng = 0
            for ng in ngrams:
                if ng in embedded_ngrams:
                    count_ng += 1
                    vectors[index, :] += w2v_ngrams[ng]
            if count_ng > 0:
                count_absent += 1
                vectors[index, :] /= count_ng
            else:
                count_absent_small += 1

    print("embeddings created, found {} known words, "
          "{} unknown words with known ngrams, "
          "{} unknown words with no known ngrams".format(count_present, count_absent, count_absent_small))

    # Saving
    np.save(filename_save, vectors)


def get_char_ngrams(word, min_n, max_n, start_char, end_char):
    """
    extract the ngrams of character from a word given min size and max_size of wanted ngrams
    :param word: word from which we want to extract char ngrams
    :param min_n: min size for ngrams (usually 3)
    :param max_n: max size for ngrams (usually 6)
    :param start_char: char to define beginning of word (usually '<')
    :param end_char: char to define end of word (usually '>')
    :return: list of string containing all the ngrams
    """
    ngrams = []
    word = start_char + word + end_char
    if len(word) < min_n:
        return ngrams

    for i in range(min_n, max_n + 1):
        for j in range(len(word) - i + 1):
            ngrams.append(word[j:j + i])

    return ngrams

def get_word_preprocessing(vocab_words=None, lowercase=True, allow_unk=True, max_length=None):

    def preproc(words):
        if max_length is not None:
            if len(words) > max_length:
                words = words[:max_length]
            else:
                for i in range(max_length - len(words)):
                    words.append(PAD)
        if lowercase:
            words = [w.lower() for w in words]
        if vocab_words is not None:
            for i in range(len(words)):
                if words[i] in vocab_words:
                    words[i] = vocab_words[words[i]]
                else:
                    if allow_unk:
                        words[i] = vocab_words[PAD]
                    else:
                        raise Exception("unknown key are not allowed ({}), check vocabulary file".format(words[i]))

        return words

    return preproc


def get_classes_preprocessing(vocab_classes=None):

    def prep(topic):
        if vocab_classes is not None:
            if topic in vocab_classes:
                topic = vocab_classes[topic]
            else:
                raise Exception("class {} does not exists".format(topic))
        return topic

    return prep
