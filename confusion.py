# -*- coding: utf-8 -*-

import os
import sys
import getopt
import numpy as np
from models.classifier import TextClassifier
from build_data import TRAIN_OUTPUT_NAME


def usage():
    print("python confusion.py -h <help?> -v <verbose?> -s <soft?> -m <mode> "
          "-f <filters (comma sep)> -n <number_filters> -p <path> -b <batch_size>")


def main():

    filters = [3, 4, 5]
    number_filters = 100
    mode = "trainable"

    directory = ""
    verbose = False
    soft = False
    batch_size = 50

    # necessary options to create model but not for inference
    init_lr = 0
    dropout_keep = 1

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hvsm:f:n:p:b:",
                                   ["help",
                                    "verbose",
                                    "soft",
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
        elif o in ("-p", "--path"):
            directory = os.path.expanduser(a)
        elif o in ("-b", "--batch_size"):
            batch_size = int(a)
        else:
            assert False, "unhandled option"

    path_trainset = os.path.join(directory, TRAIN_OUTPUT_NAME)
    if not os.path.exists(path_trainset):
        print("testset wasn't found. looked for {}".format(path_trainset))
        sys.exit()
    classifier = TextClassifier(directory, filters, number_filters,
                                dropout_keep, batch_size, init_lr, verbose, mode)
    classifier.restore_model()
    trainset = np.load(path_trainset)

    cm = classifier.confusion_matrix(trainset)

    # TODO : save the confusion matrix ? as image ? np array?


if __name__ == "__main__":
    main()
    # import cProfile
    # cProfile.run(main())

