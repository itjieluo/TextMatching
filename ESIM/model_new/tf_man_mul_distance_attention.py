import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')
from data_utils import data_utils
import params
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

data_pk = '../data_utils/char_data_old.pk'
with open(data_pk, 'rb') as f:
    train_set, test_set, word2id, id2word = pickle.load(f)

# embeddings_shape:[词表长度,200]
# embeddings = data_utils.random_embedding(id2word, params.embedding_dim)
fname = '../data_utils/word2vec_model.pkl'
embeddings = data_utils.word2vec_embedding_big_corpus(word2id, params.embedding_dim, fname)


graph_mhd = tf.Graph()
with graph_mhd.as_default():
    """
    构建神经网络的结构、损失、优化方法和评估方法
    """
    # shape[batch_size, 15]
    left_input = tf.placeholder(tf.int32, shape=[None, params.max_sentence_length], name="left_input")
    # shape[batch_size, 15]
    right_input = tf.placeholder(tf.int32, shape=[None, params.max_sentence_length], name="right_input")

    # shape[batch_size, labels]
    labels = tf.placeholder(tf.float32, shape=[None, 2], name="labels")

    # dropout keep_prob
    dropout_pl = tf.placeholder(dtype=tf.float32, shape=(), name="dropout")

    with tf.variable_scope("embeddings"):  # 命名空间
        _word_embeddings = tf.Variable(embeddings,  # shape[len_words,300]
                                       dtype=tf.float32,
                                       trainable=True,  # 嵌入层是否可以训练
                                       name="embedding_matrix")
        left_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=left_input, name="left_embeddings")
        # left_embeddings_shape：[batchsize,15,200]
        right_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=right_input, name="right_embeddings")

        left_embeddings = tf.nn.dropout(left_embeddings, dropout_pl)
        right_embeddings = tf.nn.dropout(right_embeddings, dropout_pl)

    with tf.variable_scope("one_layer_bi-lstm"):
        # 词1层bi-lstm
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=50)
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=50)
        (left_output_fw_seq, left_output_bw_seq), left_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                                left_embeddings,
                                                                                                dtype=tf.float32)
        # left_output_fw_seq_shape:[batchsize,15,50]
        # left_states_shape:[2,batchsize,50]
        left_bi_output = tf.concat([left_states[0].h, left_states[1].h], axis=-1)

        (right_output_fw_seq, right_output_bw_seq), right_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                                   right_embeddings,
                                                                                                   dtype=tf.float32)
        right_bi_output = tf.concat([right_states[0].h, right_states[1].h], axis=-1)

    with tf.variable_scope("Attention"):#命名空间
        #在词Embeding层上添加attention层
        # [[1,1],[2,2]]
        # [[0.1,0.1],[0.5,0.5]]
        attention_w = tf.get_variable('attention_omega', [params.embedding_dim, 1])
        attention_b = tf.get_variable('attention_b', [1,params.max_sentence_length])
        attention_left = tf.reduce_sum(left_embeddings * tf.expand_dims(tf.nn.softmax(tf.tanh(tf.add(
            tf.reshape(tf.matmul(tf.reshape(left_embeddings, [-1, params.embedding_dim]), attention_w),
            [-1, params.max_sentence_length]),attention_b))), -1),axis = 1)
        attention_right = tf.reduce_sum(right_embeddings * tf.expand_dims(tf.nn.softmax(tf.tanh(tf.add(
            tf.reshape(tf.matmul(tf.reshape(right_embeddings, [-1, params.embedding_dim]), attention_w),[-1, params.max_sentence_length]),
            attention_b))), -1),axis = 1)

    with tf.variable_scope("Similarity_calculation_layer"):
        def cosine_dist(input1,input2):
            pooled_len_1 = tf.sqrt(tf.reduce_sum(input1 * input1, 1))
            pooled_len_2 = tf.sqrt(tf.reduce_sum(input2 * input2, 1))
            pooled_mul_12 = tf.reduce_sum(input1 * input2, 1)
            score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 +1e-8, name="scores")
            return score

        def manhattan_dist(input1,input2):
            score = tf.exp(-tf.reduce_sum(tf.abs(input1-input2), 1))
            return score
        def multiply(input1,input2):
            score = tf.multiply(input1, input2)  # 矩阵点乘（内积）
            #tf.matmul(matrix3, matrix2)  # 矩阵相乘
            return score
        def subtract(input1,input2):
            score = tf.abs(input1-input2)
            return score


        s1_last = tf.concat([left_bi_output, attention_left], 1)
        s2_last = tf.concat([right_bi_output, attention_right], 1)

        cos = cosine_dist(s1_last, s2_last)
        man = manhattan_dist(s1_last, s2_last)
        mul = multiply(s1_last, s2_last)
        sub = subtract(s1_last, s2_last)

        # 曼哈顿距离
        # output = tf.expand_dims(manhattan_dist(left_bi_output, right_bi_output), -1)
        last_list_layer = tf.concat([mul, sub], 1)
        last_drop = tf.nn.dropout(last_list_layer, 0.8)
        dense_layer1 = tf.layers.dense(last_drop, 16, activation=tf.nn.relu)
        dense_layer2 = tf.layers.dense(last_drop, 24, activation=tf.nn.sigmoid)
        output = tf.concat([dense_layer1, dense_layer2, tf.expand_dims(cos, -1), tf.expand_dims(man, -1)], 1)


    with tf.variable_scope("classification"):
        # logits:shape[batch_size, labels]
        output = tf.layers.dense(output, 32)
        logits = tf.layers.dense(output, 2)

    # 计算损失
    with tf.variable_scope("loss"):
        logits__ = tf.nn.softmax(logits)
        ''' 标签:[[0,1],
                  [0,1],
                  [1,0]]
            预测值:[[0.3,0.7],
                    [0.2,0.8],
                    [0.6,0.4]]
            交叉熵损失

        '''
        loss = -tf.reduce_mean(0.25*labels[:,0]*tf.log(logits__[:,0])+\
                              0.75*labels[:,1]*tf.log(logits__[:,1]))



    #选择优化器
    with tf.variable_scope("train_step"):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        global_add = global_step.assign_add(1)#用于计数
        train_op = tf.train.AdamOptimizer(params.lr).minimize(loss)

    # 准确率/f1/p/r计算
    with tf.variable_scope("evaluation"):
        true = tf.cast(tf.argmax(labels, axis=-1), tf.float32)  # 真实序列的值
        labels_softmax = tf.nn.softmax(logits)
        labels_softmax_ = tf.argmax(labels_softmax, axis=-1)
        pred = tf.cast(labels_softmax_, tf.float32)  # 预测序列的值
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32),name="accuracy")
        # 混淆矩阵
        # _|0 |1 |
        # 0|2 |3 |
        # 1|2 |3 |
        # [[2,3],[2,3]]

        epsilon = 1e-7
        cm = tf.contrib.metrics.confusion_matrix(true, pred, num_classes=2)

        precision = tf.cast(cm[1][1] / tf.reduce_sum(cm[:,1]),tf.float32,name="precision")
        recall = tf.cast(cm[1][1] / tf.reduce_sum(cm[1],axis=0),tf.float32,name="recall")
        f1_score = tf.cast((2 * precision * recall / (precision + recall + epsilon)),tf.float32,name="f1_score")



with tf.Session(graph=graph_mhd) as sess:
    if params.isTrain:
        saver = tf.train.Saver(tf.global_variables())
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(params.epoch_num):
            for s1, s2, y in data_utils.get_batch(train_set, params.batch_size, shuffle=True):
                _, l, acc, p, r, f,global_nums,cmm = sess.run(
                    [train_op, loss,accuracy,precision,recall,f1_score,global_add,cm], {
                    left_input: s1,
                    right_input: s2,
                    labels : y,
                    dropout_pl: params.dropout
                })

                if global_nums % 50 == 0:
                    print(cmm)
                    # saver.save(sess, '../model_save/model.ckpt', global_step=global_nums)
                    print(
                        'train: epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4} , precision {}, recall: {:.4}, fbeta_score: {:.4} '.format(epoch , global_nums,
                                                                                          l, acc,p, r, f))


                if global_nums % 200 == 0:
                    print('-----------------valudation---------------')
                    s1, s2, y = next(data_utils.get_batch(test_set, np.shape(test_set)[0],  shuffle=True))
                    l, acc, p, r, f, global_nums,cmm = sess.run(
                        [ loss, accuracy, precision, recall, f1_score, global_add,cm], {
                            left_input: s1,
                            right_input: s2,
                            labels: y,
                            dropout_pl: params.dropout
                        })
                    print(cmm)
                    print(
                        'valudation: epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4} , precision {}, recall: {:.4}, fbeta_score: {:.4} '.format(
                            epoch, global_nums,
                            l, acc, p, r, f))
                    print('-----------------valudation---------------')
        s1, s2, y = next(data_utils.get_batch(test_set, np.shape(test_set)[0], shuffle=True))
        l, acc, p, r, f, global_nums, cmm = sess.run(
                    [loss, accuracy, precision, recall, f1_score, global_add, cm], {
                        left_input: s1,
                        right_input: s2,
                        labels: y,
                        dropout_pl: params.dropout
                    })
        print(cmm)
        print('test: loss: {:.4}, accuracy: {:.4} , precision {}, recall: {:.4}, fbeta_score: {:.4} '.format(
                        l, acc, p, r, f))
