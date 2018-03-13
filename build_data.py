# -*- coding: utf-8 -*-

import codecs
import os
import sys
import getopt
import numpy as np

from preprocessing.preprocessor import Preprocessor, load_vocabulary, get_vocabulary, write_vocabulary, \
    create_embeddings, get_classes_preprocessing, get_word_preprocessing, PAD

TRAIN_NAME = 'trainingset.txt'
TEST_NAME = 'testset.txt'
TRAIN_OUTPUT_NAME = 'train.npy'
TEST_OUTPUT_NAME = 'test.npy'
VOCABULARY_NAME = 'vocabulary.txt'
CLASSES_NAME = 'classes.txt'
IDS_OUTPUT_NAME = 'test-ids.txt'

EMBEDDINGS_INPUT_NAME = 'embeddings.sg.ns.bin'
EMBEDDINGS_INPUT_NAME_NG = 'embeddings.sg.ns.ngrams.bin'
EMBEDDINGS_OUTPUT_NAME = 'embeddings.npy'


def usage():
    print("python build_data.py -h <help?> -v <verbose?> -s <max_sentence_length> "
          "-l <lowercase?> -i <input_dir> -o <output_dir>")


def main():

    input_dir = ""
    output_dir = ""
    max_length_sentence = 100
    lc = False
    verbose = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hvls:i:o:",
                                   ["help",
                                    "verbose",
                                    "lowercase",
                                    "max_sen_length=",
                                    "input_dir=",
                                    "output_dir="
                                    ])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(2)

    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-v", "--verbose"):
            verbose = True
        elif o in ("-s", "--max_sen_length"):
            max_length_sentence = int(a)
        elif o in ("-l", "--lowercase"):
            lc = True
        elif o in ("-i", "--input_dir"):
            input_dir = os.path.expanduser(a)
        elif o in ("-o", "--output_dir"):
            output_dir = os.path.expanduser(a)
        else:
            assert False, "unhandled option"

    if not os.path.exists(input_dir):
        print("input directory does not exists... exiting")
        sys.exit()

    if output_dir == "":
        output_dir = input_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("directory {} created".format(output_dir))

    path_trainset = os.path.join(input_dir, TRAIN_NAME)
    path_testset = os.path.join(input_dir, TEST_NAME)

    if not os.path.exists(path_trainset):
        print("training set file is absent ({})".format(path_trainset))
        sys.exit()
    if not os.path.exists(path_testset):
        print("test set file is absent ({})".format(path_testset))
        sys.exit()

    path_save_voc_w = os.path.join(output_dir, VOCABULARY_NAME)
    path_save_voc_c = os.path.join(output_dir, CLASSES_NAME)

    prep_train = Preprocessor(path_trainset)
    prep_test = Preprocessor(path_testset, train=False)
    size_train = len(prep_train)
    size_test = len(prep_test)
    if verbose:
        print("Starting pre-processing on files of {} sentences".format(size_train + size_test))
    voc_w, voc_c, max_length = get_vocabulary([prep_train, prep_test], verbose)
    voc_w.add(PAD)
    write_vocabulary(voc_w, path_save_voc_w)
    write_vocabulary(voc_c, path_save_voc_c)
    embeddings_size = len(voc_w)
    del voc_w, voc_c, prep_train

    vocab_words = load_vocabulary(path_save_voc_w)

    # Loading vocabularies as dictionary
    if verbose:
        print("\nvocabulary loaded back ... {} words "
              "(might be different from before due to utf-8 encoding issues...)".format(len(vocab_words)))
    vocab_classes = load_vocabulary(path_save_voc_c)
    max_length = min(max_length, max_length_sentence)
    processing_words = get_word_preprocessing(vocab_words, max_length=max_length)
    processing_class = get_classes_preprocessing(vocab_classes)
    prep_to_int_train = Preprocessor(path_trainset,
                                     processing_words=processing_words,
                                     processing_class=processing_class)
    prep_to_int_test = Preprocessor(path_testset, train=False,
                                    processing_words=processing_words,
                                    processing_class=processing_class)
    train, _ = fill_matrix(size_train, max_length, prep_to_int_train, train=True)
    test, ids_test = fill_matrix(size_test, max_length, prep_to_int_test, train=False)
    np.save(os.path.join(output_dir, TRAIN_OUTPUT_NAME), train)
    np.save(os.path.join(output_dir, TEST_OUTPUT_NAME), test)
    with codecs.open(os.path.join(output_dir, IDS_OUTPUT_NAME), "w", encoding='utf-8') as f:
        f.write("\n".join(ids_test))

    path_w2v = os.path.join(input_dir, EMBEDDINGS_INPUT_NAME)
    path_ngrams_w2v = os.path.join(input_dir, EMBEDDINGS_INPUT_NAME_NG)
    path_save_embeddings = os.path.join(output_dir, EMBEDDINGS_OUTPUT_NAME)
    min_n = 3
    max_n = 6
    create_embeddings(vocab_words, embeddings_size, path_w2v, path_ngrams_w2v, path_save_embeddings, min_n, max_n)


def fill_matrix(size, max_length, prep, train=True):

    ids = []
    i = 0
    if train:
        x = np.zeros((size, max_length + 1))
    else:
        x = np.zeros((size, max_length))
    for id, topic, words in prep:
        ids.append(id)
        if train:
            x[i, :] = words + [topic]
        else:
            x[i, :] = words
        i += 1
    return x, ids


if __name__ == "__main__":
    main()
