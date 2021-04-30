import numpy as np
import pickle
import os
import sys
sys.path.append(os.getcwd())
from src.layers import *
from src.preprocess import *


class FastText:
    def __init__(self, V, D, O):            # O is the number of classes
        W_in = np.sqrt(2/(V+D))*np.random.uniform(low = -1, high = 1, size=(V, D))
        W_out = np.sqrt(2/(D+O))*np.random.uniform(low = -1, high = 1, size = (D, O))
        b_out = np.sqrt(1/O)*np.random.uniform(low = -1, high = 1, size = (1, O))

        self.params =[]
        self.grads =[]
        self.embedding = Embedding(W_in)
        self.output_layer = Affine(W_out, b_out)
        self.layer = [self.embedding, self.output_layer]
        for layer in self.layer:
            self.params += layer.params
            self.grads += layer.grads
        self.loss_layer = SoftmaxWithLoss()

        self.cache = None

    def predict(self, xs):
        sen_len = len(xs)

        xs = self.embedding.forward(xs)
        xs_norm = np.linalg.norm(xs, axis=1, keepdims=True)
        xs /= xs_norm
        xs = np.sum(xs, axis=0, keepdims=True) / sen_len
        out = self.output_layer.forward(xs)
        self.cache = (sen_len, xs_norm)
        return out

    def forward(self, xs, ts):
        out = self.predict(xs)
        loss = self.loss_layer.forward(out, ts)
        return loss
        
    def backward(self, lr):
        sen_len, xs_norm = self.cache
        dout= self.loss_layer.backward()
        dout = self.output_layer.backward(dout,lr)
        dx = dout/sen_len
        dx = np.tile(dx, (sen_len, 1))/xs_norm
        self.embedding.backward(dx, lr)
        return None

