# -*- coding: utf-8 -*-

import os
import sys
import getopt
import codecs
import numpy as np
from preprocessing.preprocessor import load_classes
from models.classifier import TextClassifier
from build_data import TEST_OUTPUT_NAME, TEST_NAME, CLASSES_NAME


def usage():
    print("python evaluate.py -h <help?> -v <verbose?> -s <soft?> -m <mode> -t <topn> "
          "-f <filters (comma sep)> -n <number_filters> -p <path> -b <batch_size>")


def main():

    filters = [3, 4, 5]
    number_filters = 100
    topn = 0
    mode = "trainable"

    directory = ""
    verbose = False
    soft = False
    batch_size = 50

    # necessary options to create model but not for inference
    init_lr = 0
    dropout_keep = 1

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hvst:m:f:n:p:b:",
                                   ["help",
                                    "verbose",
                                    "soft",
                                    "topn=",
                                    "mode=",
                                    "filters=",
                                    "number_filters=",
                                    "path=",
                                    "batch_size=",
                                    ])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit()

    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-v", "--verbose"):
            verbose = True
        elif o in ('-s', "--soft"):
            soft = True
        elif o in ("-m", "--mode"):
            mode = a
        elif o in ("-f", "--filters"):
            filters = [int(elt.strip()) for elt in a.split(',')]
        elif o in ("-n", "--number_filters"):
            number_filters = int(a)
        elif o in ("-t", "--topn"):
            topn = int(a)
        elif o in ("-p", "--path"):
            directory = os.path.expanduser(a)
        elif o in ("-b", "--batch_size"):
            batch_size = int(a)
        else:
            assert False, "unhandled option"

    path_testset = os.path.join(directory, TEST_OUTPUT_NAME)
    if not os.path.exists(path_testset):
        print("testset wasn't found. looked for {}".format(path_testset))
        sys.exit()
    classifier = TextClassifier(directory, filters, number_filters,
                                dropout_keep, batch_size, init_lr, verbose, mode)
    classifier.restore_model()
    testset = np.load(path_testset)

    scores = classifier.evaluate(testset, soft)
    if soft:
        scores_name = "scores.{}.{}.f{}.nf{}.npy".format("soft",
                                                         mode,
                                                         "-".join([str(elt) for elt in filters]),
                                                         str(number_filters))
    else:
        scores_name = "scores.{}.f{}.nf{}.npy".format(mode,
                                                      "-".join([str(elt) for elt in filters]),
                                                      str(number_filters))
    path_scores = os.path.join(directory, "scores")
    if not os.path.exists(path_scores):
        os.makedirs(path_scores)
    path_scores_current = os.path.join(path_scores, scores_name)
    np.save(path_scores_current, scores)
    if verbose:
        print("scores saved under name {} in {}".format(scores_name, path_scores))
    if topn > 0 and soft:
        print("saving {} best(s) docs by class".format(topn))
        path_samples = os.path.join(directory, "samples")
        sample_name = "sample.{}.f{}.nf{}.txt".format(mode,
                                                      "-".join([str(elt) for elt in filters]),
                                                      str(number_filters))
        if not os.path.exists(path_samples):
            os.makedirs(path_samples)
        path_samples_current = os.path.join(path_samples, sample_name)
        classes = load_classes(os.path.join(directory, CLASSES_NAME))
        with codecs.open(os.path.join(directory, TEST_NAME), "r", encoding='utf-8') as f:
            testset_text = f.read().splitlines()
        with codecs.open(path_samples_current, "w", encoding='utf-8') as f:
            for i in range(scores.shape[1]):
                f.write("##### - {} - #####\n\n\n".format(classes[i]))
                bests = np.argsort(scores[:, i])[:topn]
                for j in bests:
                    f.write(testset_text[j])
                    f.write("\n\n")


if __name__ == "__main__":
    main()
    # import cProfile
    # cProfile.run(main())
