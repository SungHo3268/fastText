import numpy as np
import os
import sys
sys.path.append(os.getcwd())

def validation(test_text, test_label, model):
    count = 0
    for i in range(len(test_text)):
        y = np.argmax(model.predict(test_text[i]), axis=1)  # softmax 까지 안해도 됨,
        if y == (test_label[i]):
            count += 1
    return count/len(test_text)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x = x - np.max(x, axis=1 , keepdims=True)
    exp_x = np.exp(x)
    sum_x = np.sum(exp_x, axis=1, keepdims=True)

    return exp_x / sum_x


def cross_entropy_loss(x, t):
    if x.ndim == 1:
        t = t.reshape(1, t.size)
        x = x.reshape(1, x.size)

    # if t is one-hot encoded.
    if x.size == t.size:
        t = t.argmax(axis = 1)

    batch_size = x.shape[0]
    return -np.sum(np.log(z[np.arange(batch_size), t] + 1e-07)) / batch_size