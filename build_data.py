# -*- coding: utf-8 -*-

import codecs
import os
import sys

from preprocessing.preprocessor import Preprocessor, load_vocabulary, get_vocabulary, write_vocabulary, \
    create_embeddings, get_classes_preprocessing, get_word_preprocessing, PAD

dirname_context = os.path.expanduser('~/AppData/Local/ProtoStudio/')
dirname_save = os.path.expanduser('~/Documents/Projects/Classification/')
max_length_sentence = 100

if len(sys.argv) < 3:
    print("missing at least one parameter. usage : python build_data.py <context> <lang>")
    sys.exit(2)

context = sys.argv[1]
lang = sys.argv[2]
extension = None
if len(sys.argv) > 3:
    extension = sys.argv[3]

dirname_context = os.path.join(dirname_context, context, lang, "Classification")
if not os.path.exists(dirname_context):
    print("directory does not exists, lang ({}) or context ({}) might be misspelled"
          " or Classification folder might be absent".format(lang, context))
    sys.exit(2)

if extension is not None:
    dirname_save = os.path.join(dirname_save, context, extension)
else:
    dirname_save = os.path.join(dirname_save, context)
if not os.path.exists(dirname_save):
    os.makedirs(dirname_save)
    print("directory {} was created".format(dirname_save))

if extension is not None:
    path_training_set = os.path.join(dirname_context, "trainingset.{}.txt".format(extension))
else:
    path_training_set = os.path.join(dirname_context, "trainingset.txt")

if not os.path.exists(path_training_set):
    print("training set file is absent ({})".format(path_training_set))
    sys.exit(2)

prep_train = Preprocessor(path_training_set)
# TODO : use testset as well to extract all vocabulary ?

# Creating vocabulary and saving
size = len(prep_train)
voc_w, voc_c, max_length = get_vocabulary([prep_train])
path_save_voc_w = os.path.join(dirname_save, "vocabulary.txt")
path_save_voc_c = os.path.join(dirname_save, "classes.txt")
voc_w.add(PAD)
write_vocabulary(voc_w, path_save_voc_w)
write_vocabulary(voc_c, path_save_voc_c)

embeddings_size = len(voc_w)
del voc_w, voc_c

# Loading vocabularies as dictionary
vocab_words = load_vocabulary(path_save_voc_w)
print("\nvocabulary loaded back ... {} words "
      "(might be different from before due to utf-8 encoding problems...)".format(len(vocab_words)))
vocab_classes = load_vocabulary(path_save_voc_c)

max_length = min(max_length, max_length_sentence)
processing_words = get_word_preprocessing(vocab_words, max_length=max_length)
processing_class = get_classes_preprocessing(vocab_classes)

prep_to_int = Preprocessor(path_training_set, processing_words=processing_words, processing_class=processing_class)
path_save_train_int = os.path.join(dirname_save, "training.int.txt")
with codecs.open(path_save_train_int, "w", encoding='utf-8') as f:
    f.write("{}\t{}\n".format(size, max_length + 1))
    for _, topic, words in prep_to_int:
        f.write(" ".join([str(elt) for elt in words]))
        f.write(" ")
        f.write(str(topic))
        f.write('\n')

path_w2v = os.path.join(dirname_context, "..", "embeddings.sg.ns.bin")
path_ngrams_w2v = os.path.join(dirname_context, "..", "embeddings.sg.ns.ngrams.bin")
path_save_embeddings = os.path.join(dirname_save, "embeddings.npy")
min_n = 3
max_n = 6
create_embeddings(vocab_words, embeddings_size, path_w2v, path_ngrams_w2v, path_save_embeddings, min_n, max_n)
