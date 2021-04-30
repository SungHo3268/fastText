import numpy as np


def sigmoid(xs):
    out = 1./(1.+np.exp(-xs))
    return out

def softmax(xs):
    out = 0
    if xs.ndim == 2:
        xs -= xs.max(axis=1, keepdims=True)
        xs = np.exp(xs)
        sum_xs = np.sum(xs, axis=1, keepdims=True)
        out = xs / sum_xs

    elif xs.ndim == 1:
        xs -= xs.max()
        xs = np.exp(xs)
        out = xs / np.sum(xs)
    return out


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    # if t is represented by one-hot-vectors, make it argument-represent.
    if t.ndim == 2 and y.ndim == t.ndim:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-07)) / batch_size


def remove_duplicate(params, grads):
    params, grads = params[:], grads[:]  # copy list
    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 가중치 공유 시
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 경사를 더함
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 가중치를 전치행렬로 공유하는 경우(weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                        params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg:
                    break
            if find_flg:
                break

        if not find_flg:
            break
    return params, grads


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate
    return None
