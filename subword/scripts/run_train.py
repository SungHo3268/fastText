import argparse
from distutils.util import strtobool as _bool
from torch.utils.tensorboard import SummaryWriter
import pickle
import gzip
from tqdm.auto import tqdm
import time
import json
import os
import sys
sys.path.append(os.getcwd())
from src.Huffman import *
from src.utils import *
from src.models import *
from src.functions import *


################################ Arg parser ################################
parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--model', type=str, default='SkipGram')
parser.add_argument('--activation', type=str, default='NEG')
parser.add_argument('--least_freq', type=int, default=5)
parser.add_argument('--window_size', type=int, default=5)
parser.add_argument('--negative', type=int, default=5)
parser.add_argument('--max_epoch', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.025)
parser.add_argument('--sub_t', type=float, default=1e-05)
parser.add_argument('--hidden_dim', type=int, default=300)
parser.add_argument('--eval_interval', type=int, default=10000)
parser.add_argument('--min_n', type=int, default=3, help="at least 3")
parser.add_argument('--max_n', type=int, default=6, help="at most 6")

args = parser.parse_args()
log_dir = 'log/{}_{}{}_subword{}to{}_t{}_dim{}_{}epoch/'\
    .format(args.model, args.activation, args.negative, args.min_n, args.max_n, args.sub_t, args.hidden_dim, args.max_epoch)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
with open(log_dir + 'argparse.json', 'w') as f:
    json.dump(args.__dict__, f)


############################## tensor board ##################################
tb_dir = os.path.join(log_dir, 'tb')
ckpt_dir = os.path.join(log_dir, 'ckpt')

if not os.path.exists(log_dir):
    os.mkdir(log_dir)
if not os.path.exists(tb_dir):
    os.mkdir(tb_dir)
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

tb_writer = SummaryWriter(tb_dir)


############################## Init net #################################
# load dictionary
file = 'datasets/preprocessed/wordDict.pkl'
word_to_id, id_to_word, vocabulary = load_dict(file)


# load (subword + word) dictionary
file = 'datasets/preprocessed/subwordDict.pkl'
subword_to_id, id_to_subword = load_subDict(file)
V = len(subword_to_id)


# word_indexing lookup table
word_to_subidx, id_to_subidx = word_indexing(word_to_id, subword_to_id, args.min_n, args.max_n)


# total sentence num
'''sen_num = 0
for i in tqdm(range(5), desc='count the total number of sentences.'):
    if i < 10: num = '0'+ str(i)
    else: num = str(i)
    file = 'datasets/preprocessed/sentence_list_excepted/sentence_list_{}.pkl'.format(i)
    with open(file, 'rb') as fr:
        sentence_list = pickle.load(fr)
        sen_num += len(sentence_list)'''
sen_num = 30301028


# probability of subsampling
id_to_sub_p = {}
if args.sub_t > 0:
    id_to_sub_p = subsampling_prob(vocabulary, word_to_id, args.sub_t)


# UniGramSampling Table
UnigramTable=[]
pos_index = []
if args.activation == 'NEG':
    UnigramTable, pos_index = unigramTable(vocabulary, word_to_id)
    UnigramTable = np.array(UnigramTable)


############################### start train ##################################
# select the model
model = SkipGram(V, args.hidden_dim, word_to_id, id_to_word, subword_to_id, id_to_subidx, UnigramTable, args.negative)

# training start
real_train_word = 0
current = 0
total_loss = 0
loss_count = 0

time_list = []
start_t = time.time()
for epoch in range(args.max_epoch):
    print("epoch: %d/%d" %(epoch+1, args.max_epoch))
    # per dataset segment       -- save weight per segment.
    for j in range(5):
        with open('datasets/preprocessed/sentence_list/sentence_list_{}.pkl'.format(j), 'rb') as fr:
            sentence_list = pickle.load(fr)
        # per sentence
        for i in tqdm(range(len(sentence_list)), desc='dataset '+str(j+1)+'/5', bar_format="{l_bar}{bar:20}{r_bar}"):
            current += 1
            # Do subsampling.        return a sentence which is string, not id.
            if args.sub_t > 0:
                sentence = subsampling(sentence_list[i], id_to_sub_p)
                if not sentence:         # if all words discarded.
                    continue
            else: sentence = sentence_list[i]
            # per center word
            for c, center in enumerate(sentence):
                real_train_word += 1

                # apply decreasing learning rate
                alpha = 1 - current/(sen_num*args.max_epoch)
                if alpha <= 0.0001:
                    alpha = 0.0001
                lr = args.lr * alpha

                # make contexts by shrunk args.window_size.
                contexts = make_contexts(args.window_size, sentence, c)
                if not contexts:
                    continue

                # start real training.
                loss= model.forward(center, contexts)
                model.backward(lr)      # it includes update the gradient to parameters.
                total_loss += loss
                loss_count += 1

                if (args.eval_interval is not None) and (real_train_word%args.eval_interval ==1):
                    avg_loss = total_loss/loss_count

                    total_loss = 0
                    loss_count = 0

                    tb_writer.add_scalar('loss/real_train_word(*{})'.format(args.eval_interval), avg_loss, real_train_word)
                    tb_writer.add_scalar('lr/real_train_word(*{})'.format(args.eval_interval), lr, real_train_word)
                tb_writer.flush()
        # save the temp weight.
        weight_dir = os.path.join(log_dir, 'weight')
        if not os.path.exists(weight_dir):
            os.mkdir(weight_dir)
        with open(os.path.join(weight_dir, 'ckpt_weight.pkl'), 'wb') as fw:
            pickle.dump((model.W_in, model.W_out), fw)

        et = (time.time() - start_t)/3600
        time_list.append(et)
        print("epoch: {}/{}, dataset {}, elapsed_time: {}[h]".format(epoch+1, args.max_epoch, str(j+1)+'/5', et))
        start_t = time.time()

    print("real train words per epoch: ", real_train_word)
    # save the weights
    weight_dir = os.path.join(log_dir, 'weight')
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    with open(os.path.join(weight_dir, 'weight_{}epoch.pkl'
            .format(epoch+1)), 'wb') as fw:
        pickle.dump((model.W_in, model.W_out), fw)

total_time = 0
for t in time_list:
    total_time += t
print("total_time: {}[h]".format(total_time))