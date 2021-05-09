from collections import Counter
import os
import sys
sys.path.append(os.getcwd())
from src.functions import *


# def make_words(file):
#     with open(file, 'r', encoding='utf8') as f:
#         data = f.read()
#     word_list = data.split()
#     return word_list


def make_words(file):
    word_list = []
    with open(file) as infile:
        for line in tqdm(infile):
            word_list.append(line.split())
    return word_list


def make_wordDict(file, least_freq=5):
    print("Please wait... loading dataset..")
    word_list = make_words(file)
    counter = Counter(word_list)
    print("Complete!")

    word_freq = Counter()
    for word in tqdm(counter.keys(), desc='counting the frequency of words', bar_format='{l_bar}{bar:20}{r_bar}'):
        if counter[word] >= least_freq:
            word_freq[word] = counter[word]

    vocabulary = dict(word_freq.most_common())
    word_to_id = {}
    id_to_word = {}
    for word in tqdm(vocabulary.keys(), desc='making dictionary', bar_format='{l_bar}{bar:20}{r_bar}'):
        word_to_id[word] = len(word_to_id)
        id_to_word[len(id_to_word)] = word
    return word_to_id, id_to_word, vocabulary


def make_subwordDict(vocabulary, min_n, max_n):
    subword_to_id = {'[pad]': 0}
    id_to_subword = {0: '[pad]'}
    ngram_set = set()
    for word in tqdm(vocabulary.keys(), desc='making subword dictionary', bar_format='{l_bar}{bar:20}{r_bar}'):
        if word == '[pad]':
            continue
        word = '<' + word + '>'
        subword_to_id[word] = len(subword_to_id)
        id_to_subword[len(id_to_subword)] = word

        wl = len(word)
        for i in range(wl):
            if wl-i < min_n:
                break
            end = min(max_n, wl-i)
            for j in range(min_n-1, end):       # index 는 길이보다 한 개 빼줘야 함.
                if j+1 == wl:                   # word itself 는 위에서 미리 처리했음.
                    continue
                ngram_set.add(word[i: i+j+1])

    # convert set 'ngrams' to dictionary
    for ngram in ngram_set:
        subword_to_id[ngram] = len(subword_to_id)
        id_to_subword[len(id_to_subword)] = ngram
    return subword_to_id, id_to_subword


def make_corpus(file, word_to_id, sub_p):
    with open(file, 'rb') as fr:
        word_list = pickle.load(fr)
    corpus = []                 # (#tokens, )
    for word in word_list:
        if word not in word_to_id:
            continue
        idx = word_to_id[word]

        # subsampling
        if sub_p[idx] < np.random.random():
            corpus.append(idx)
    return np.array(corpus)
