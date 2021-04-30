import numpy as np
from tqdm import tqdm
import pandas as pd
import collections
import pickle
import os
import sys
from src.utils import *
sys.path.append(os.getcwd())


def make_dict(path, ngrams=False):
    # train.csv
    content = []
    label = []
    with open(path + 'train.csv', 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    for line in tqdm(lines, desc="load (train) data"):
        label.append(int(clean(line[:4])))
        content.append(clean(line[4:]))
    word_to_id = {"UNK" : 0}
    id_to_word = {0: "UNK"}
    for line in tqdm(content, desc='make dict'):
        line = line.split()
        if ngrams:
            line += ['{}_{}'.format(line[idx], line[idx+1]) for idx in range(len(line)-1)]
        for word in line:
            if word not in word_to_id:
                word_to_id[word] = len(word_to_id)
                id_to_word[len(id_to_word)] = word

    if ngrams:
        with open(path + "bi_dict.pkl", "wb") as fw:
            pickle.dump((word_to_id, id_to_word), fw)
    else:
        with open(path + "uni_dict.pkl", "wb") as fw:
            pickle.dump((word_to_id, id_to_word), fw)
    return word_to_id
    


def make_train_corpus(path, word_to_id, ngrams=False):
    content = []
    label = []
    text = []
    max_len = 0
    with open(path + 'train.csv', 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    for line in tqdm(lines, desc="load train data"):
        label.append(int(clean(line[:4])))
        content.append(clean(line[4:]))
    for line in tqdm(content, desc='make train corpus'):
        line = line.split()
        if ngrams:
            line += ['{}_{}'.format(line[idx], line[idx+1]) for idx in range(len(line)-1)]
        temp = []
        for word in line:
            if word in word_to_id:
                temp.append(word_to_id[word])
        text.append(temp)
        if len(temp) > max_len:
            max_len = len(temp)
    for i in tqdm(range(len(text)), desc='zero padding'):
        if len(text[i]) < max_len:
            for _ in range(max_len - len(text[i])):
                text[i].append(0)
            
    if ngrams:
        with open(path + "bi_train_corpus.pkl", "wb") as fw:
            pickle.dump((text, label), fw)
    else:
        with open(path + "uni_train_corpus.pkl", "wb") as fw:
            pickle.dump((text, label), fw)
    return None


def make_test_corpus(path, word_to_id, ngrams=False):
    content = []
    label = []
    text = []
    max_len = 0
    with open(path + 'test.csv', 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    for line in tqdm(lines, desc="load test data"):
        label.append(int(clean(line[:4])))
        content.append(clean(line[4:]))
    for line in tqdm(content, desc='make test corpus'):
        line = line.split()
        if ngrams:
            line += ['{}_{}'.format(line[idx], line[idx+1]) for idx in range(len(line)-1)]
        temp = []
        for word in line:
            if word in word_to_id:
                temp.append(word_to_id[word])
        text.append(temp)
        if len(temp) > max_len:
            max_len = len(temp)
    for i in tqdm(range(len(text)), desc='zero padding'):
        if len(text[i]) < max_len:
            for _ in range(max_len - len(text[i])):
                text[i].append(0)

    if ngrams:
        with open(path + "bi_test_corpus.pkl", "wb") as fw:
            pickle.dump((text, label), fw)
    else:
        with open(path + "uni_test_corpus.pkl", "wb") as fw:
            pickle.dump((text, label), fw)
    return None
