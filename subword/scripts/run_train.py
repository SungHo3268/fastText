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
parser.add_argument('--max_epoch', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--sub_t', type=float, default=1e-04)
parser.add_argument('--hidden_dim', type=int, default=300)
parser.add_argument('--eval_interval', type=int, default=10000)
parser.add_argument('--min_n', type=int, default=3, help="at least 3")
parser.add_argument('--max_n', type=int, default=6, help="at most 6")
parser.add_argument('--data_seg', type=int, default=5)
parser.add_argument('--grad_div', type=int, default=64)
args = parser.parse_args()

log_dir = f'log/subword{args.min_n}to{args.max_n}_{args.max_epoch}epoch_{args.grad_div}div/'
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
file = 'datasets/wiki/preprocessed/wordDict.pkl'
word_to_id, id_to_word, vocabulary = load_dict(file)


# load (subword + word) dictionary
file = 'datasets/wiki/preprocessed/subwordDict.pkl'
subword_to_id, id_to_subword = load_subDict(file)
V = len(subword_to_id)


# word_indexing lookup table
word_to_subidx, id_to_subidx = word_indexing(word_to_id, subword_to_id, args.min_n, args.max_n)


# UniGramSampling Table
UnigramTable = []
if args.activation == 'NEG':
    UnigramTable = unigramTable(vocabulary, word_to_id)


# probability of subsampling
sub_p = None
if args.sub_t > 0:
    sub_p = subsampling_prob(vocabulary, args.sub_t)


# load corpus
with open('datasets/wiki/preprocessed/corpus/corpus_all.pkl', 'rb') as fr:
    corpus = pickle.load(fr)
token_num = len(corpus)    # 123353509
print("The total tokens in corpus: ", token_num)
all_contexts = make_contexts(args.window_size, corpus)
sub_corpus, sub_contexts = subsampling(corpus, all_contexts, sub_p)     # after subsampling real training token_num
real_token_num = len(sub_corpus)
print('total training tokens: ', real_token_num)
print('\n')


# select the model
model = SkipGram(V, args.hidden_dim, id_to_subidx, UnigramTable, args.negative)


############################### start train ##################################
# training start
tokens = 0              # real train token num
seg_count = 0           # at [0, args.data_seg]
total_loss = 0
pos_loss = 0
neg_loss = 0
loss_count = 0
seg_time_list = []          # elapsed time per segment
epoch_time_list = []        # elapsed time per epoch
start_t = time.time()
for epoch in range(args.max_epoch):
    print("epoch: %d/%d" % (epoch+1, args.max_epoch))
    # shuffle the data
    print("Shuffling the data....")
    rand_idx = np.random.permutation(len(sub_corpus))
    rand_corpus = sub_corpus[rand_idx]
    rand_contexts = sub_contexts[rand_idx]
    seg_corpus, seg_contexts = split_seg(rand_corpus, rand_contexts, args.data_seg)
    print("complete..!")

    # per dataset segment       -- save weight per segment.
    for corp, cont in zip(seg_corpus, seg_contexts):
        seg_count += 1
        # per word
        for c in tqdm(range(len(corp)),
                      desc='corpus_seg {}/{}'.format(seg_count-(args.data_seg*epoch), args.data_seg),
                      bar_format="{l_bar}{bar:20}{r_bar}"):
            tokens += 1
            # apply decreasing learning rate
            alpha = 1 - (tokens / (real_token_num*args.max_epoch))
            if alpha <= 0.0001:
                alpha = 0.0001
            lr = args.lr * alpha

            # start model training.
            loss, pos, neg = model.forward(corp[c], cont[c])
            model.backward(lr, args.grad_div)      # it includes update the gradient to parameters.
            total_loss += loss
            pos_loss += pos
            neg_loss += neg
            loss_count += 1

            if (args.eval_interval is not None) and (tokens % args.eval_interval == 1):
                avg_loss = total_loss/loss_count
                avg_pos_loss = pos_loss/loss_count
                avg_neg_loss = neg_loss/loss_count
                total_loss = 0
                pos_loss = 0
                neg_loss = 0
                loss_count = 0

                tb_writer.add_scalar('total_loss/tokens'.format(args.eval_interval), avg_loss, tokens)
                tb_writer.add_scalar('pos_loss/tokens'.format(args.eval_interval), avg_pos_loss, tokens)
                tb_writer.add_scalar('neg_loss/tokens'.format(args.eval_interval), avg_neg_loss, tokens)
                tb_writer.add_scalar('lr/tokens'.format(args.eval_interval), lr, tokens)
            tb_writer.flush()

        # save the temp weight.
        weight_dir = os.path.join(log_dir, 'weight')
        if not os.path.exists(weight_dir):
            os.mkdir(weight_dir)
        with open(os.path.join(weight_dir, 'weight_seg{}.pkl'.format(seg_count)), 'wb') as fw:
            pickle.dump((model.W_in, model.W_out), fw)

        seg_t = (time.time() - start_t)/3600
        seg_time_list.append(seg_t)
        print("corpus_seg {}/{}, elapsed_time: {}[h]".format(
            seg_count-(args.data_seg*epoch), args.data_seg, seg_t))
        start_t = time.time()

    epoch_time_list.append(sum(seg_time_list))
    print("train time at {}epoch: ".format(epoch+1), sum(seg_time_list))
    print('\n')
    seg_time_list = []
print("total_time: {}[h]".format(sum(epoch_time_list)))
