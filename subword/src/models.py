import numpy as np
import os
import sys
import time
import pickle
sys.path.append(os.getcwd())
from src.utils import *
from src.functions import *


class SkipGram:
    def __init__(self, V, D, word_to_id, id_to_word, subword_to_id, id_to_subidx, UnigramTable, negative):
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.subword_to_id = subword_to_id
        self.id_to_subidx = id_to_subidx
        self.UnigramTable = UnigramTable
        self.negative = negative

        self.W_in = 0.01*np.random.randn(V, D).astype('f')
        self.W_out = np.zeros((V, D)).astype('f')
        self.cache = None

    def forward(self, center, contexts):
        cache = []
        loss = 0
        for context in contexts:
            #get negative samples
            neg_sample = []
            while 1:
                b = np.random.randint(low=0, high=len(self.UnigramTable), size=self.negative)
                if context in self.UnigramTable[b]:
                    continue
                else:
                    neg_sample = self.UnigramTable[b]
                    break
            target = np.append(context, neg_sample)
            # forward
            x, x_idx = getWordVec(center, self.W_in, self.id_to_subidx)       # x=(#sub, D), x_idx=(#sub, )
            t = self.W_out[target]              # t=(#target, D)
            score = np.sum(np.dot(x, t.T), axis=0)            # out = (#target, )
            out = sigmoid(np.array(score))
            p_loss = -np.log(out[0] + 1e-07)
            n_loss = -np.sum(np.log(1-out[1:] + 1e-07))             # loss = (1, )
            loss += (p_loss + n_loss)
            cache.append((x, x_idx, t, target, out))
        self.cache = cache
        return float(loss)/len(contexts)

    def backward(self, lr):
        for x, x_idx, t, target, out in self.cache:
            '''
            x = (#sub, D)
            x_idx = (#sub, )
            t = (#target, D)
            target = (#target, )
            out = (#target, )
            '''
            # backward
            dout = out.copy()                               # dout = (#target, )
            dout[0] -= 1                                    # subtract positive label=1
            dout = np.tile(dout, (len(x_idx), 1))           # dout = (#sub, #target)
            dW_out = np.dot(x.T, dout).T                    # dW_out = (#target, D)
            dx = np.dot(dout, t)                            # dx = (#sub, D)

            # clip_grads(dW_out, max_norm=5)
            # clip_grads(dx, max_norm=5)

            #update
            self.W_out[target] -= dW_out/64 * lr
            self.W_in[x_idx] -= dx/64 *lr
