import argparse
from distutils.util import strtobool as _bool
from torch.utils.tensorboard import SummaryWriter
import json
import numpy as np
from tqdm.auto import tqdm
import time
import pickle
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.getcwd())
from src.models import *
from src.functions import *

############################################# Argparse #############################################
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='server')
parser.add_argument('--dataset', type=int, default=5, help="0:AG | 1:Sogou | 2:Dbpedia | 3: YelpP | \
                                                        4:YelpF | 5:Yahoo | 6:AmazonP | 7:AmazonF")
parser.add_argument('--ngrams', type=str, default='uni', help="uni | bi")
parser.add_argument('--dim', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.15)
parser.add_argument('--max_epoch', type=int, default=10)

args = parser.parse_args()
data = {0:'AG', 1:'Sogou', 2:'Dbpedia', 3:'YelpP', 4:'YelpF', 5:'Yahoo', 6:'AmazonP', 7:'AmazonF'}


############################################# Setting #############################################
data_list = {0:'datasets/ag_news_csv/',
             1:'datasets/sogou_news_csv/',
             2:'datasets/dbpedia_csv/',
             3:'datasets/yelp_review_polarity_csv/',
             4:'datasets/yelp_review_full_csv/',
             5:'datasets/yahoo_answers_csv/',
             6:'datasets/amazon_review_polarity_csv/',
             7:'datasets/amazon_review_full_csv/'}

path = data_list[args.dataset]
# load dictionary
with open(path+'{}_dict.pkl'.format(args.ngrams), 'rb') as fr:
    word_to_id, id_to_word = pickle.load(fr)

# load train dataset
with open(path+'{}_train_corpus.pkl'.format(args.ngrams), 'rb') as fr:
    train_text, train_label = pickle.load(fr)       # notice, raw train_label starts at 1
train_text = np.array(train_text)
train_label = np.array(train_label) - 1

# load test dataset
with open(path+'{}_test_corpus.pkl'.format(args.ngrams), 'rb') as fr:
    test_text, test_label = pickle.load(fr)
test_text = np.array(test_text)
test_label = np.array(test_label) - 1

# model
V = len(word_to_id)
D = args.dim
O = len(set(train_label))
model = FastText(V, D, O)

# train
loss_list = [1]
acc_list = [0]
sen_num = len(train_text)
start_t = time.time()
print('{}_{}_dim{}_e{}_lr{}/'.format(data[args.dataset], args.ngrams, args.dim, args.max_epoch, args.lr))
for epoch in range(args.max_epoch):
    rand_idx = np.random.permutation(sen_num)
    train_text = train_text[rand_idx]
    train_label = train_label[rand_idx]
    acc = 0
    loss = 0
    for i in tqdm(range(len(train_text))):
        # SGD optimizer with linearly decaying.
        alpha = 1 - i/sen_num
        if alpha <= 0.0001:
            alpha = 0.0001
        lr = args.lr * alpha

        #AdaGrad optimizer.
        # lr = args.lr
        
        loss += model.forward(train_text[i], train_label[i])
        model.backward(lr)
    loss /= len(train_text)
    loss_list.append(loss)

    rand_idx = np.random.permutation(len(test_text))
    test_text = test_text[rand_idx]
    test_label = test_label[rand_idx]
    acc = validation(test_text, test_label, model)
    acc_list.append(acc)
    
    print("epoch: {}/{}  |  loss: {}  |  acc: {}[%] |  elapsed_t: {}[s]"
          .format(epoch+1, args.max_epoch, loss, acc*100, time.time()-start_t))
print('\n\n')


