import re
import json
import jieba
import pickle
import numpy as np
import pandas as pd
from gensim.models import word2vec

# 产生随机的embedding矩阵
def random_embedding(id2word, embedding_dim):
    """

    :param id2word:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(id2word), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


# 通过word2vec拿到我们所需embedding矩阵
def word2vec_embedding_big_corpus(word2id, embedding_dim, fname):
    """

    :param id2word:
    :param embedding_dim:
    :param fname:
    :return:
    """
    embedding_dim = 200  # 这边腾讯的embedding都是200维，所以自己填充的也是200维。
    model = word2vec.Word2Vec.load(fname)
    embedding_mat = np.zeros((len(word2id), 200))
    for word, idex in word2id.items():
        try:
            embedding_mat[idex] = model[word]
        except:
            # 随机初始化向量
            random_num = np.random.uniform(-0.25, 0.25, (embedding_dim))
            embedding_mat[idex] = random_num
    return embedding_mat

# 将数据pad，生成batch数据返回，这里没有取余数。
def get_batch(data, batch_size, shuffle=False):
    """
    :param data:
    :param batch_size:
    :param shuffle:
    :return:
    """
    # 乱序没有加
    if shuffle:
        np.random.shuffle(data)
    for i in range(len(data) // batch_size):
        data_size = data[i * batch_size: (i + 1) * batch_size]
        s1_data, s2_data, label_data = [], [], []
        for (s1_set, s2_set, y_set) in data_size:
            s1_data.append(s1_set)
            s2_data.append(s2_set)
            label_data.append(y_set)
        yield np.array(s1_data), np.array(s2_data), np.array(label_data)
    if len(data)%batch_size != 0:
        s1_data, s2_data, label_data = [], [], []
        data_size = data[-batch_size:]
        for (s1_set, s2_set, y_set) in data_size:
            s1_data.append(s1_set)
            s2_data.append(s2_set)
            label_data.append(y_set)
        yield np.array(s1_data), np.array(s2_data), np.array(label_data)

# 将数据pad，生成batch数据返回，这里没有取余数。
def get_batch_arc(data, batch_size, shuffle=False):
    """
    :param data:
    :param batch_size:
    :param shuffle:
    :return:
    """
    # 乱序没有加
    if shuffle:
        np.random.shuffle(data)
    for i in range(len(data) // batch_size):
        data_size = data[i * batch_size: (i + 1) * batch_size]
        s1_data, label_data = [], []
        for (s1_set, y_set) in data_size:
            s1_data.append(s1_set)
            label_data.append(y_set)
        yield np.array(s1_data),  np.array(label_data)



# 将数据pad，生成batch数据返回，这里没有取余数。
def get_batch_all(data, batch_size, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param shuffle:
    :return:
    """
    # 乱序没有加
    if shuffle:
        np.random.shuffle(data)
    for i in range(len(data) // batch_size):
        data_size = data[i * batch_size: (i + 1) * batch_size]
        s1_data, s2_data, c1_data, c2_data, label_data = [], [], [], [], []
        for (s1_set, s2_set,c1_set, c2_set, y_c_set) in data_size:
            s1_data.append(s1_set)
            s2_data.append(s2_set)
            c1_data.append(c1_set)
            c2_data.append(c2_set)
            label_data.append(y_c_set)
        yield np.array(s1_data), np.array(s2_data), np.array(c1_data), np.array(c2_data), np.array(label_data)

