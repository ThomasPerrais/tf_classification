# -*- coding: utf-8 -*-

import os
import sys
import getopt
from models.classifier import TextClassifier


def usage():
    print("python train.py -h <help?> -v <verbose?> -m <mode> -e <epochs> -f <filters (comma sep)> "
          "-n <number_filters> -p <path> -b <batch_size> -l <learning_rate> -d <dropout_keep_proba>")


def main():

    filters = [3, 4, 5]
    number_filters = 100
    batch_size = 50
    init_lr = 0.025
    dropout_keep = 0.5
    epochs = 100
    directory = ""
    verbose = False
    mode = "trainable"

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hvm:e:f:n:p:b:l:d:",
                                   ["help",
                                    "verbose",
                                    "mode="
                                    "epochs",
                                    "filters=",
                                    "number_filters=",
                                    "path=",
                                    "batch_size=",
                                    "lr=",
                                    "dropout="
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
        elif o in ("-m", "--mode"):
            mode = a
        elif o in ("-n", "--number_filters"):
            number_filters = int(a)
        elif o in ("-f", "--filters"):
            filters = [int(elt.trim()) for elt in a.split(',')]
        elif o in ("-p", "--path"):
            directory = os.path.expanduser(a)
        elif o in ("-b", "--batch_size"):
            batch_size = int(a)
        elif o in ("-l", "--lr"):
            init_lr = float(a)
        elif o in ("-d", "--dropout"):
            dropout_keep = float(a)
        elif o in ("-e", "--epochs"):
            epochs = int(a)
        else:
            assert False, "unhandled option"

    classifier = TextClassifier(directory, filters, number_filters,
                                dropout_keep, batch_size, init_lr, verbose, mode)
    classifier.train_model(epochs)


if __name__ == "__main__":
    main()
    # import cProfile
    # cProfile.run(main())
