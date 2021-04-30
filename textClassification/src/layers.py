import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from src.utils import *
from src.functions import *


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, x):
        W, = self.params
        output = W[x]
        self.idx = x
        return output

    def backward(self, dout, lr):
        W,  = self.params
        W[self.idx] -= lr *dout
        # self.grads[0] = dout*dout
        # W[self.idx] -= dout*lr / (np.sqrt(self.grads[0]) + 1e-07)

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.cache = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.cache= x
        return out

    def backward(self, dout, lr):
        W, b = self.params
        x = self.cache
        dx = np.matmul(dout, W.T)
        dW = np.matmul(x.T, dout)
        db = np.sum(dout, axis=0)

        # update
        # self.grads[0] += dW * dW
        # self.grads[1] += db * db
        # W -= lr*dW / (np.sqrt(self.grads[0]) + 1e-07)
        # b -= lr*db / (np.sqrt(self.grads[1]) + 1e-07)

        W -= lr*dW
        b -= lr*db
        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.params = None
        self.grads = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = sigmoid(x)
        batch = t.shape[0]
        loss = -np.sum(t*np.log(self.y + 1e-07) + (1-t)*np.log(1-self.y + 1e-07))
        return loss/batch

    def backward(self):
        dx = self.y - self.t
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params = None
        self.grads = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = softmax(x)       # (1, class_num)
        self.t = t                # (1, )
        loss = -np.sum(np.log(self.y[0, t] + 1e-07))
        return loss

    def backward(self):
        self.y[0, self.t] -= 1
        return self.y
    

class Softmax:
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sum_dx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sum_dx
        return dx


class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None

    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dout):
        dx = self.out*(1. - self.out) * dout
        return dx
