import pickle
import gzip
import numpy as np
from tqdm.auto import tqdm


def load_dict(file):
    with open(file, 'rb') as fr:
        word_to_id, id_to_word, vocabulary = pickle.load(fr)
    print("vocabulary size: ", len(vocabulary))
    return word_to_id, id_to_word, vocabulary


def load_subDict(file):
    with open(file, 'rb') as fr:
        subword_to_id, id_to_subword = pickle.load(fr)
    print("subword vocabulary size: ", len(subword_to_id))
    return subword_to_id, id_to_subword


def unigramTable(vocabulary, word_to_id):
    UnigramTable = []   # negative sampling pool = Unigram table.       len(UnigramTable) = total_token_num = total_freq
    for word in tqdm(vocabulary.keys(), desc='Making UnigramTable', bar_format="{l_bar}{bar:20}{r_bar}"):
        # freq = int(pow(vocabulary[word], 3/4))
        freq = int(np.sqrt(vocabulary[word]))
        UnigramTable += ([word_to_id[word]] * freq)
    print("UnigramTable length applied 1/2 pow: ", len(UnigramTable))
    return np.array(UnigramTable)


def subsampling_prob(vocabulary, sub_t):
    sub_p = []
    freq = np.array(list(vocabulary.values()))
    prob = freq/sum(freq)
    for p in tqdm(prob, desc='get subsampling prob: ', bar_format="{l_bar}{bar:20}{r_bar}"):
        sub_p.append(1-np.sqrt(sub_t/p))              # probability to be discarded (paper)
        # sub_p.append((1+np.sqrt(p/sub_t))*sub_t/p)  # probability to be kept (public code)
    return np.array(sub_p)


def subsampling(corpus, contexts, sub_p):
    sub_corpus = []
    sub_contexts = []
    for i in tqdm(range(len(corpus)), desc="Subsampling...", bar_format="{l_bar}{bar:20}{r_bar}"):
        # probability to be discarded
        if sub_p[corpus[i]] < np.random.random():
            sub_corpus.append(corpus[i])
            sub_contexts.append(contexts[i])
        '''# probability to be kept
        if sub_p[corpus[i]] > np.random.random():
            sub_corpus.append(corpus[i])
            sub_contexts.append(contexts[i])'''
    return np.array(sub_corpus), np.array(sub_contexts, dtype=object)


def word_indexing(word_to_id, subword_to_id, min_n, max_n):
    word_to_subidx = {}
    id_to_subidx = {}
    for word, index in tqdm(word_to_id.items(), desc='make word_index lookup dictionary',
                            bar_format="{l_bar}{bar:20}{r_bar}"):
        idx = []
        word_ = '<' + word + '>'
        if (len(word_) > max_n) or (len(word_) < min_n):               # 6 보다 긴 단어는 word itself vector 를 포함하지 않으므로.
            idx.append(subword_to_id[word_])
        wl = len(word_)
        for i in range(wl):
            if wl-i < min_n:
                break
            end = min(max_n, wl-i)
            for j in range(min_n-1, end):       # index 는 길이보다 한 개 빼줘야 함.
                idx.append(subword_to_id[word_[i: i+j+1]])
        word_to_subidx[word] = idx
        id_to_subidx[index] = idx
    return word_to_subidx, id_to_subidx


def split_seg(corpus, contexts, seg):
    seg_len = len(corpus) // seg + 1
    seg_corpus = []
    seg_contexts = []
    for i in range(seg):
        cor_segment = corpus[i*seg_len: (i+1)*seg_len]
        con_segment = contexts[i*seg_len: (i+1)*seg_len]
        seg_corpus.append(cor_segment)
        seg_contexts.append(con_segment)
    return seg_corpus, seg_contexts


def make_contexts(window_size, corpus):        # c means 'current sentence position'
    # make random window_size
    b = int(np.random.randint(1, window_size+1))
    cl = len(corpus)
    # make contexts by shrunk args.window_size.
    all_contexts = []
    for c in tqdm(range(cl), desc='make contexts...', bar_format='{l_bar}{bar:20}{r_bar}'):
        contexts = []
        for j in range(-b, b+1):
            # only take account into within boundaries.
            cur = c+j
            if cur < 0:
                continue
            elif cur == c:     # if j==0
                continue
            elif cur >= cl:
                break
            else:       # cur(=current_index) is among sentence.
                contexts.append(corpus[cur])         # complete contexts
        all_contexts.append(contexts)
    return all_contexts


def getWordVec(target, subword_vectors, id_to_subidx):      # target = (#target, ) ~= (6,)
    sub_indices = []
    word_vectors = []
    for word in target:
        sub_index = id_to_subidx[word]              # sub_index = (#sub, )
        sub_vector = subword_vectors[sub_index]     # sub_vector = (#sub, D)
        sub_indices.append(sub_index)
        word_vectors.append(sub_vector.sum(axis=0))
    return np.array(word_vectors), sub_indices


def getSubWordVec(center, subword_vectors, id_to_subidx):       # center word = (1, )
    sub_index = id_to_subidx[center]
    sub_vector = subword_vectors[sub_index]
    return sub_vector, np.array(sub_index)


########################################################################################################
##########################################  Analogy functions ##########################################
########################################################################################################
def sub_to_wordVec(subword_vectors, word_to_subidx):
    current = 0
    word_vectors = np.zeros((len(word_to_subidx), subword_vectors.shape[1]))
    for sub_idx in tqdm(word_to_subidx.values(), desc='get word vectors', bar_format="{l_bar}{bar:20}{r_bar}"):
        word_vector = subword_vectors[sub_idx]
        word_vectors[current] = np.sum(word_vector, axis=0)
        current += 1
    return word_vectors


def checkValid(question, word_to_id):
    valid_que = []
    invalid_que = []
    for que in question:
        if (que[0] in word_to_id) and (que[1] in word_to_id) and (que[2] in word_to_id) and (que[3] in word_to_id):
            valid_que.append(que)
        else:
            if que[3] in word_to_id:
                invalid_que.append(que)
    return valid_que, invalid_que


def convert2vec(valid, word_vectors, word_to_id):
    a = []
    b = []
    c = []
    for s in valid:
        a_vec = word_vectors[word_to_id[s[0]]]
        b_vec = word_vectors[word_to_id[s[1]]]
        c_vec = word_vectors[word_to_id[s[2]]]

        a_norm = np.linalg.norm(a_vec)
        b_norm = np.linalg.norm(b_vec)
        c_norm = np.linalg.norm(c_vec)

        a.append(a_vec / a_norm)
        b.append(b_vec / b_norm)
        c.append(c_vec / c_norm)
    return np.array(a), np.array(b), np.array(c)


def cos_similarity(predict, word_vectors):
    norm_predict = np.linalg.norm(predict, axis=1)
    norm_words = np.linalg.norm(word_vectors, axis=1)

    similarity = np.dot(predict, word_vectors.T)      # similarity = (N, V)
    similarity *= 1/norm_words
    similarity = similarity.T
    similarity *= 1/norm_predict
    similarity = similarity.T

    return similarity


def count_in_top4(similarity, id_to_word, valid):
    count = 0
    current = -1
    for sim in tqdm(similarity, desc='Assessing accuracy...', bar_format="{l_bar}{bar:20}{r_bar}"):     # sim=(V, )
        current += 1
        asc_arg = np.argsort(sim)[::-1]         # ascending sorting
        top4 = list(asc_arg[:4])
        for j in range(len(top4)):
            pred = id_to_word[top4[j]]
            if pred in valid[current]:
                if pred == valid[current][-1]:
                    count += 1
            else:
                break
    return count
