# -*- coding: utf-8 -*-

import codecs
import os
import sys
import getopt

from preprocessing.preprocessor import Preprocessor, load_vocabulary, get_vocabulary, write_vocabulary, \
    create_embeddings, get_classes_preprocessing, get_word_preprocessing, PAD

TRAIN_NAME = 'trainingset.txt'
EMBEDDINGS_NAME = 'embeddings.sg.ns.bin'


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

    path_training_set = os.path.join(input_dir, TRAIN_NAME)
    if not os.path.exists(path_training_set):
        print("training set file is absent ({})".format(path_training_set))
        sys.exit()

    prep_train = Preprocessor(path_training_set)
    size = len(prep_train)
    if verbose:
        print("Starting preprocessing on file of {} sentences".format(size))
    voc_w, voc_c, max_length = get_vocabulary([prep_train], verbose)
    path_save_voc_w = os.path.join(output_dir, "vocabulary.txt")
    path_save_voc_c = os.path.join(output_dir, "classes.txt")
    voc_w.add(PAD)
    write_vocabulary(voc_w, path_save_voc_w)
    write_vocabulary(voc_c, path_save_voc_c)

    embeddings_size = len(voc_w)
    del voc_w, voc_c, prep_train

    # Loading vocabularies as dictionary
    vocab_words = load_vocabulary(path_save_voc_w)
    if verbose:
        print("\nvocabulary loaded back ... {} words "
              "(might be different from before due to utf-8 encoding issues...)".format(len(vocab_words)))
    vocab_classes = load_vocabulary(path_save_voc_c)

    max_length = min(max_length, max_length_sentence)
    processing_words = get_word_preprocessing(vocab_words, max_length=max_length)
    processing_class = get_classes_preprocessing(vocab_classes)

    prep_to_int = Preprocessor(path_training_set, processing_words=processing_words, processing_class=processing_class)
    path_save_train_int = os.path.join(output_dir, "training.int.txt")
    with codecs.open(path_save_train_int, "w", encoding='utf-8') as f:
        f.write("{}\t{}\n".format(size, max_length + 1))
        for _, topic, words in prep_to_int:
            f.write(" ".join([str(elt) for elt in words]))
            f.write(" ")
            f.write(str(topic))
            f.write('\n')

    path_w2v = os.path.join(input_dir, "embeddings.sg.ns.bin")
    path_ngrams_w2v = os.path.join(input_dir, "embeddings.sg.ns.ngrams.bin")
    path_save_embeddings = os.path.join(output_dir, "embeddings.npy")
    min_n = 3
    max_n = 6
    create_embeddings(vocab_words, embeddings_size, path_w2v, path_ngrams_w2v, path_save_embeddings, min_n, max_n)

if __name__ == "__main__":
    main()
