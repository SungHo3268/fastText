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
    print("subword dictionary size: ", len(subword_to_id))
    return subword_to_id, id_to_subword


def load_data(file):
    with gzip.open(file, 'rb') as fr:
        sentence_list = pickle.load(fr)
    print("load training dataset.")
    return sentence_list


def subsampling_prob(vocabulary, word_to_id, sub_t):
    key = []
    idx = []
    prob = []
    sub_p = []
    total_freq = 0
    for freq in tqdm(vocabulary.values(), desc='count words', bar_format="{l_bar}{bar:20}{r_bar}"):
        total_freq += freq

    for (word, f) in vocabulary.items():
        key.append(word)
        idx.append(word_to_id[word])
        prob.append(f/total_freq)
    for p in prob:
        if p==0:
            sub_p.append(0)
            continue
        sub_p.append((1+np.sqrt(p/sub_t)) * sub_t/p)
    id_to_sub_p = dict(np.stack((idx, sub_p), axis=1))
    return id_to_sub_p


def unigramTable(vocabulary, word_to_id):
    UnigramTable = []   # sampling pool = Unigram table.       len(UnigramTable) = total_word
    current = 0         # the current index of sub_list
    pos_index = []      # the start index of words with word(char) in UnigramTable.
    for word in tqdm(vocabulary.keys(), desc='Making UnigramTable', bar_format="{l_bar}{bar:20}{r_bar}"):
        if word == '\s':
            continue
        freq = int(pow(vocabulary[word], 3/4))
        pos_index.append((current, current+freq))   # (start_idx, next_start_idx)
        current += freq         # It's for the next word's index
        temp = [word_to_id[word]] * freq
        UnigramTable.extend(temp)
    print("UnigramTable length applied 3/4 pow: ", len(UnigramTable))
    return UnigramTable, pos_index


def subsampling(sentence_, id_to_sub_p):
    sentence = []
    for word in sentence_:
        # if word not in word_to_id.keys():         # it's a double check after checking in make_sentence_list
        #         continue
        if id_to_sub_p[word] > np.random.random():
            sentence.append(word)
    return sentence


def make_contexts(window_size, sentence, c):        # c means 'current sentence position'
    # make random window_size
    b = int(np.random.randint(1, window_size+1))

    # make contexts by shrunk args.window_size.
    contexts = []
    for j in range(-b, b+1):
        # only take account into within boundaries.
        cur = c+j
        if cur < 0:
            continue
        elif cur == c:     # if j==0
            continue
        elif cur >= len(sentence):
            break
        else:       # cur(=current_index) is among sentence.
            if sentence[cur] == 0:
                continue
            else:
                contexts.append(sentence[cur])         # complete contexts
    return contexts


def code_to_id(codes, root, vocabulary):
    node = root
    idx = []
    code_sign = []
    for word in tqdm(vocabulary, desc='get codes to index', bar_format="{l_bar}{bar:20}{r_bar}"):
        temp0 = []
        temp1 = []
        code = codes[word]
        for c in code:
            if c == '0':
                temp0.append(node.index)
                temp1.append(-1)
                node = node.left
            elif c == '1':
                temp0.append(node.index)
                temp1.append(1)
                node = node.right
            if node.index is None:
                node = root
                break
        idx.append(temp0)
        code_sign.append(temp1)
    return idx, code_sign


def word_indexing(word_to_id, subword_to_id, min_n, max_n):
    word_to_subidx = {}
    id_to_subidx = {}
    for word, index in tqdm(word_to_id.items(), desc='make word_index lookup dictionary', bar_format="{l_bar}{bar:20}{r_bar}"):
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
                idx.append(subword_to_id[word_[i : i+j+1]])
        word_to_subidx[word] = idx
        id_to_subidx[index] = idx
    return word_to_subidx, id_to_subidx


def getWordVec(words, subword_vectors, id_to_subidx):
    sub_indices = []
    word_vectors = []
    if type(words) == int or type(words) == np.int64:
        sub_index = id_to_subidx[words]
        sub_vector = subword_vectors[sub_index]
        return sub_vector, sub_index
    else:
        for word in words:
            sub_index = id_to_subidx[word]
            sub_vector = subword_vectors[sub_index]
            sub_indices.append(sub_index)
            word_vectors.append(sub_vector)
        return word_vectors, sub_indices


def getOOV(word, subword_vectors, subword_to_id, min_n, max_n):
    word = '<' + word + '>'
    wl = len(word)
    sub_vec = 0
    count = 0
    for i in range(wl):
        if wl-i < min_n:
            break
        end = min(max_n, wl-i)
        for j in range(min_n-1, end):       # index 는 길이보다 한 개 빼줘야 함.
            subword = word[i: i+j+1]
            if subword in subword_to_id:
                sub_id = subword_to_id[subword]
                sub_vec += subword_vectors[sub_id]
                count += 1

    if count != 0:
        sub_vec /= count
    else:
        print("There's no subword about '{}'".format(word))
    return sub_vec


####################################################################
########################  Analogy functions ########################
####################################################################
def checkValid(question, vocabulary):
    valid_que = []
    invalid_que = []
    for que in question:
        if que[0] in vocabulary and que[1] in vocabulary and que[2] in vocabulary and que[3] in vocabulary:
            valid_que.append(que)
        else:
            if que[3] in vocabulary:
                invalid_que.append(que)
    return valid_que, invalid_que


def convert2vec(valid, word_vectors, subword_vectors, word_to_id, subword_to_id, min_n, max_n):
    a = []
    b = []
    c = []
    d = []
    no_subword = []
    for i, s in enumerate(valid):
        if s[0] not in word_to_id:
            a_vec = getOOV(s[0], subword_vectors, subword_to_id, min_n, max_n)
            if type(a_vec) == int:      # if there is no subword in subword dictionary.
                no_subword.append(i)
                continue
        else: a_vec = word_vectors[word_to_id[s[0]]]

        if s[1] not in word_to_id:
            b_vec = getOOV(s[1], subword_vectors, subword_to_id, min_n, max_n)
            if type(b_vec) == int:
                no_subword.append(i)
                continue
        else: b_vec = word_vectors[word_to_id[s[1]]]

        if s[2] not in word_to_id:
            c_vec = getOOV(s[2], subword_vectors, subword_to_id, min_n, max_n)
            if type(c_vec) == int:
                no_subword.append(i)
                continue
        else: c_vec = word_vectors[word_to_id[s[2]]]

        d_vec = word_vectors[word_to_id[s[3]]]      # d word is always in vocabulary.

        a_norm = np.linalg.norm(a_vec)
        b_norm = np.linalg.norm(b_vec)
        c_norm = np.linalg.norm(c_vec)
        d_norm = np.linalg.norm(d_vec)
        
        a.append(a_vec/a_norm)
        b.append(b_vec/b_norm)
        c.append(c_vec/c_norm)
        d.append(d_vec/d_norm)
    return np.array(a), np.array(b), np.array(c), np.array(d)


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
    max_top4 = []
    sim_top4 = []
    for i in tqdm(range(len(similarity)), bar_format="{l_bar}{bar:20}{r_bar}"):
        max_arg = np.argsort(similarity[i])[::-1]
        temp = list(max_arg[:4])
        max_top4.append(temp)
        sim_top4.append(list(similarity[i][temp]))

        for j in range(4):
            pred = id_to_word[temp[j]]
            if pred in valid[i]:
                if pred == valid[i][3]:
                    count += 1
            else: break
    return max_top4, sim_top4, count


def avgSubwordVec(subword_vectors, word_to_subidx):
    word_vectors = np.zeros((len(word_to_subidx), subword_vectors.shape[1]))
    for i, sub_idx in tqdm(enumerate(word_to_subidx.values()), desc='get word vectors', bar_format="{l_bar}{bar:20}{r_bar}"):
        word_vector = subword_vectors[sub_idx]
        word_vector = np.sum(word_vector, axis=0)
        word_vectors[i] = word_vector/len(sub_idx)
    return word_vectors
