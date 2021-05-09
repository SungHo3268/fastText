import argparse
from distutils.util import strtobool as _bool
import pandas as pd
import pickle
import numpy as np
from tqdm.auto import tqdm
import os
import sys
sys.path.append(os.getcwd())
from src.functions import *


################################ Arg parser ################################
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=int, default=5678)
parser.add_argument('--logdir', type=str, default="subword3to6_5epoch_64div")
parser.add_argument('--weight', type=str, default='weight_seg25.pkl')
parser.add_argument('--data_seg', type=int, default=10)
parser.add_argument('--min_n', type=int, default=3)
parser.add_argument('--max_n', type=int, default=6)
args = parser.parse_args()


############################### Init Net ##################################
# load analogy data
with open("datasets/wiki/analogy/questions-words.txt", 'r') as fr:
    loaded = fr.readlines()
count = 0
semantic = []
syntactic = []
for line in loaded:
    if line[0] == ':':
        count += 1
        continue
    elif line == '\n':
        continue
    if count < 6:
        semantic.append(line.split())
    else:
        syntactic.append(line.split())


'''# load author's word vectors
file = './../fastText-master/result/fil9.vec'
with open(file, 'r') as fr:
    vectors = pd.read_csv(fr)
word_vectors = []
word_to_id = {}
id_to_word = {}
for line in tqdm(vectors.values):
    line = line[0].split()
    temp = line[1:]
    word_vectors.append(temp)
    word_to_id[line[0]] = len(word_to_id)
    id_to_word[len(id_to_word)] = line[0]
word_vectors = np.array(word_vectors).astype(float)'''


# load my word_vectors
file = f'log/{args.logdir}/weight/{args.weight}'
with open(file, 'rb') as fr:
    word_vectors = pickle.load(fr)


# Check whether my word vectors contain all words in questions
valid_syn, oov_syn = checkValid(syntactic, word_to_id)
valid_sem, oov_sem = checkValid(semantic, word_to_id)
print('\n')
print("valid syntactic: %d/%d\n" % (len(valid_syn), len(syntactic)))
print("valid semantic: %d/%d" % (len(valid_sem), len(semantic)))


############################### valid evaluate ##################################
batch1 = len(valid_syn)//args.data_seg + 1
batch2 = len(valid_sem)//args.data_seg + 1
syn_counts = 0
sem_counts = 0
for i in range(args.data_seg):
    print('data: {}/{}'.format(i+1, args.data_seg))
    batch_syn = valid_syn[i*batch1: (i+1)*batch1]
    batch_sem = valid_sem[i*batch2: (i+1)*batch2]

    # syntactic
    a, b, c = convert2vec(batch_syn, word_vectors, word_to_id)
    predict_syn = b - a + c
    similarity_syn = cos_similarity(predict_syn, word_vectors)
    syn_count = count_in_top4(similarity_syn, id_to_word, batch_syn)
    syn_counts += syn_count

    # semantic
    a, b, c = convert2vec(batch_sem, word_vectors, word_to_id)
    predict_sem = b - a + c
    similarity_sem = cos_similarity(predict_sem, word_vectors)
    sem_count = count_in_top4(similarity_sem, id_to_word, batch_sem)
    sem_counts += sem_count

syn_acc = syn_counts/len(valid_syn) * 100
sem_acc = sem_counts/len(valid_sem) * 100
print('\n')
print("******************** sisg accuracy ********************")
print("syntactic acc: {:.1f} [%]".format(syn_acc))
print("semantic acc: {:.1f} [%]".format(sem_acc))
print('\n')
