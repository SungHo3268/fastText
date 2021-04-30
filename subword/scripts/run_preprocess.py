import argparse
from distutils.util import strtobool as _bool
import json
import pickle
import os
import sys
sys.path.append(os.getcwd())
from src.preprocess import *

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--f_start', type=int, default=1, help="1~99")
parser.add_argument('--f_end', type=int, default=99, help='1~99')
parser.add_argument('--least_freq', type=int, default=5)
parser.add_argument('--min_n', type=int, default=3)
parser.add_argument('--max_n', type=int, default=6)
parser.add_argument('--seg', type=int, default=5, help="How many numbers do you divide the file into?")

args = parser.parse_args()

with open('datasets/preprocessed/preprocess_argparse.json', 'w') as f:
    json.dump(args.__dict__, f)


############################################################################
# make word dictionary
word_to_id, id_to_word, vocabulary = make_wordDict(args.f_start, args.f_end, args.least_freq)
with open('datasets/preprocessed/wordDict.pkl', 'wb') as fw:
    pickle.dump((word_to_id, id_to_word, vocabulary), fw)
print("Succeed to save (word_to_id, id_to_word, vocabulary) to 'wordDict.pkl'.")
print("vocabulary length : ", len(vocabulary))


# load the dictionary for making sentence list
with open('datasets/preprocessed/wordDict.pkl', 'rb') as fr:
    word_to_id, id_to_word, vocabulary = pickle.load(fr)


# make subword dictionary which (word + subword)
subword_to_id, id_to_subword = make_subwordDict(vocabulary, args.min_n, args.max_n)
with open('datasets/preprocessed/subwordDict.pkl', 'wb') as fw:
    pickle.dump((subword_to_id, id_to_subword), fw)
print("Succeed to save (subword_to_id, id_to_subword) to 'subwordDict.pkl'.")
print("vocabulary length : ", len(subword_to_id))


# make sentence list excepted non_vocab
batch = args.f_end//args.seg + 1
for i in range(args.seg):
    if i == 0:
        sentence_list = make_sentences(word_to_id, args.f_start, batch*(i+1))
    else:
        sentence_list = make_sentences(word_to_id, batch*i, batch*(i+1))
    with open('datasets/preprocessed/sentence_list/sentence_list_{}.pkl'.format(i), 'wb') as fw:
        pickle.dump(sentence_list, fw)
    print("Succeed to make sentence list{}.".format(i))
    print("The length of the sentence list{} is ".format(i), len(sentence_list))

