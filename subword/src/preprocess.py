import pickle
import numpy as np
from collections import Counter
from tqdm.auto import tqdm
import os
import sys
sys.path.append(os.getcwd())


def make_words(file):
    with open(file, 'r', encoding='utf8') as fr:
        data = fr.read()
    word_list = []
    word = ''
    for c in data:
        if (c == '\n') or (c ==' ') or (c == '\t'):
            word_list.append(word)
            word = ''
            if c == '\n':
                word_list.append('\s')
        else: word += c
    return word_list


def make_wordDict(start, end, least_freq=5):
    counter = Counter()
    for i in tqdm(range(start, end+1)):
        if i < 10: num = '0'+ str(i)
        else: num = str(i)
        file = 'datasets/pre_train/1-billion-word/training/news.en-000' + num + '-of-00100'
        word_list = make_words(file)
        counter += Counter(word_list)

    temp = Counter()
    for word in counter.keys():
        if counter[word] < least_freq:
            temp[word] += 100          # 무조건 없어지게 큰 값(least_freq 보다 상대적으로)을 할당

    counter -= temp
    vocabulary = {'\s':0}
    vocabulary.update(dict(counter.most_common()))
    vocabulary['\s'] = 0
    word_to_id = {}
    id_to_word = {}
    for word in vocabulary.keys():
        word_to_id[word] = len(word_to_id)
        id_to_word[len(id_to_word)] = word
    return word_to_id, id_to_word, vocabulary


def make_subwordDict(vocabulary, min_n, max_n):
    subword_to_id = {}
    id_to_subword = {}
    ngram_set = set()
    for word in tqdm(vocabulary.keys()):
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
                ngram_set.add(word[i : i+j+1])

    # convert set 'ngrams' to dictionary
    for ngram in ngram_set:
        subword_to_id[ngram] = len(subword_to_id)
        id_to_subword[len(id_to_subword)] = ngram
    return subword_to_id, id_to_subword


def make_sentences(word_to_id, start, end):
    sentence_list = []
    for i in tqdm(range(start, end)):
        if i < 10: num = '0'+ str(i)
        else: num = str(i)
        file = 'datasets/pre_train/1-billion-word/training/news.en-000' + num + '-of-00100'
        word_list = make_words(file)

        sentence = []
        for word in word_list:
            if word in word_to_id.keys():
                if word == '\s':
                    sentence_list.append(sentence)
                    sentence = []
                else: sentence.append(word_to_id[word])
    return sentence_list

