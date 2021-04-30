import numpy as np
import pickle
import sys
import os
sys.path.append(os.getcwd())
from src.preprocess import *


data_list = {0:'datasets/ag_news_csv/',
             1:'datasets/sogou_news_csv/',
             2:'datasets/dbpedia_csv/',
             3:'datasets/yelp_review_polarity_csv/',
             4:'datasets/yelp_review_full_csv/',
             5:'datasets/yahoo_answers_csv/',
             6:'datasets/amazon_review_polarity_csv/',
             7:'datasets/amazon_review_full_csv/'}


i = 5

print("****************", data_list[i][9:-1], "****************")
path = data_list[i]
print("<< uni >>")
word_to_id = make_dict(path, ngrams=False)
make_train_corpus(path, word_to_id, ngrams=False)
make_test_corpus(path, word_to_id, ngrams=False)

print("<< bi >>")
word_to_id = make_dict(path, ngrams=True)
make_train_corpus(path, word_to_id, ngrams=True)
make_test_corpus(path, word_to_id, ngrams=True)
print("make uni/bi dictionary(word_to_id, id_to_word) "
      "and train/test corpus(text, label)\n\n")
