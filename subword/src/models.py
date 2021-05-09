import numpy as np
import os
import sys
import time
import pickle
sys.path.append(os.getcwd())
from src.utils import *
from src.functions import *


class SkipGram:
    def __init__(self, V, D, id_to_subidx, UnigramTable, negative):
        self.id_to_subidx = id_to_subidx
        self.UnigramTable = UnigramTable
        self.negative = negative
        # with open('log/subword3to6_5epoch_64div/weight/weight_seg15.pkl', 'rb') as fr:
        #     self.W_in, self.W_out = pickle.load(fr)
        self.W_in = 0.01*np.random.randn(V, D).astype('f')
        self.W_out = np.zeros((V, D)).astype('f')
        self.cache = None

    def forward(self, center, contexts):
        cache = []
        pos_loss = 0
        neg_loss = 0
        for context in contexts:        # 'context' is a positive target.
            # get negative samples
            neg_samples = []
            while 1:
                b = np.random.randint(low=0, high=len(self.UnigramTable), size=self.negative)
                if context in self.UnigramTable[b]:
                    continue
                else:
                    neg_samples = self.UnigramTable[b]
                    break
            target = np.append(context, neg_samples)        # target = (positive + #negative) = (1 + 5) = (6, )

            x, x_idx = getSubWordVec(center, self.W_in, self.id_to_subidx)  # x=(#sub, D), x_idx=(#sub, )
            t, t_idx = getWordVec(target, self.W_out, self.id_to_subidx)    # t=(#target, D), t_idx=(#target, #sub)
            score = np.sum(np.dot(x, t.T), axis=0)                          # score = (#target, )
            out = sigmoid(np.array(score))                                  # out = (#target, )
            pos_loss += -np.log(out[0] + 1e-07)                                # pos_loss = (1, )
            neg_loss += -np.sum(np.log(1-out[1:] + 1e-07))                     # neg_loss = (1, )
            cache.append((x, x_idx, t, t_idx, out))
        self.cache = cache
        return float(pos_loss + neg_loss)/len(contexts), float(pos_loss)/len(contexts), float(neg_loss)/len(contexts)

    def backward(self, lr, grad_div):
        for x, x_idx, t, t_idx, out in self.cache:
            """
            x = (#sub, D)
            x_idx = (#sub, )
            t = (#target, D)
            t_idx = (#target, #sub)
            out = (#target, )
            """
            # backward
            dout = out.copy()                               # dout = (#target, )
            dout[0] -= 1                                    # subtract positive label=1
            dscore = np.tile(dout, (len(x_idx), 1))         # score = (#sub, #target)
            dx = np.dot(dscore, t)                          # dx = (#sub, D)
            dW_out = np.dot(x.T, dscore).T                  # dW_out = (#target, D)

            dx /= grad_div
            dW_out /= grad_div

            # update
            for i, idx in enumerate(t_idx):                 # t_idx = (#target, #sub)       #sub은 각  target 마다 상이함.
                self.W_out[idx] -= dW_out[i] * lr
            self.W_in[x_idx] -= dx * lr
