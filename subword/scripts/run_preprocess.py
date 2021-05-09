import argparse
from distutils.util import strtobool as _bool
import json
import pickle
import os
import sys
from collections import Counter
sys.path.append(os.getcwd())
from src.preprocess import *
from src.functions import *

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--least_freq', type=int, default=5)
parser.add_argument('--min_n', type=int, default=3)
parser.add_argument('--max_n', type=int, default=6)
parser.add_argument('--seg', type=int, default=10, help="How many numbers do you divide the file into?")

args = parser.parse_args()

with open('datasets/wiki9/preprocessed/preprocess_argparse.json', 'w') as f:
    json.dump(args.__dict__, f)


############################################################################
# # make word dictionary
# word_to_id, id_to_word, vocabulary = make_wordDict(file, args.least_freq)
# with open('datasets/wiki9/preprocessed/wordDict.pkl', 'wb') as fw:
#     pickle.dump((word_to_id, id_to_word, vocabulary), fw)
# print("Succeed to save (word_to_id, id_to_word, vocabulary) to 'wordDict.pkl'.")
# print("vocabulary length : ", len(vocabulary))
with open('datasets/wiki9/preprocessed/wordDict.pkl', 'rb') as fr:
    word_to_id, id_to_word, vocabulary = pickle.load(fr)


# # make subword dictionary which (word + subword)
# subword_to_id, id_to_subword = make_subwordDict(vocabulary, args.min_n, args.max_n)
# with open('datasets/wiki9/preprocessed/subwordDict.pkl', 'wb') as fw:
#     pickle.dump((subword_to_id, id_to_subword), fw)
# print("Succeed to save (subword_to_id, id_to_subword) to 'subwordDict.pkl'.")
# print("vocabulary length : ", len(subword_to_id))
with open('datasets/wiki9/preprocessed/subwordDict.pkl', 'rb') as fr:
    subword_to_id, id_to_subword = pickle.load(fr)


# make corpus
corpus = make_corpus(file, word_to_id)
with open('datasets/wiki9/preprocessed/corpus/corpus_all.pkl', 'wb') as fw:
    pickle.dump(np.array(corpus), fw)
print("Succeed to make corpus_all.")
print("The length of corpus is ", len(corpus))
