import numpy as np
import tensorflow as tf
import sys
sys.path.append("..")
from data_utils import data_utils
import params
import pickle

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

data_pk = '../data_utils/char_data_old.pk'
with open(data_pk, 'rb') as f:
    train_data, test_data, char2id, id2char = pickle.load(f)

# embeddings = data_utils.random_embedding(char2id, params.embedding_dim)
fname = '../data_utils/word2vec_model.pkl'
embeddings = data_utils.word2vec_embedding_big_corpus(char2id, params.embedding_dim, fname)

u = 50
d = 20
graph_mhd = tf.Graph()
with graph_mhd.as_default():
    """
    构建神经网络的结构、损失、优化方法和评估方法
    """
    # shape[batch_size, sentences]
    left_input = tf.placeholder(tf.int32, shape=[None, params.max_sentence_length], name="left_input")
    # shape[batch_size, sentences]
    right_input = tf.placeholder(tf.int32, shape=[None, params.max_sentence_length], name="right_input")

    # shape[batch_size, sentences, labels]
    labels = tf.placeholder(tf.float32, shape=[None, 2], name="labels")

    # dropout keep_prob
    dropout_pl = tf.placeholder(dtype=tf.float32, shape=(), name="dropout")

    with tf.variable_scope("embeddings"):  # 命名空间
        _word_embeddings = tf.Variable(embeddings,  # shape[len_words,300]
                                       dtype=tf.float32,
                                       trainable=True,  # 嵌入层是否可以训练
                                       name="embedding_matrix")
        left_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=left_input, name="left_embeddings")
        right_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=right_input, name="right_embeddings")

        left_embeddings = tf.nn.dropout(left_embeddings, dropout_pl)
        right_embeddings = tf.nn.dropout(right_embeddings, dropout_pl)

    with tf.variable_scope("cell_by_one_layer_bi-lstm"):
        # 词1层bi-lstm
        cell_fw = tf.nn.rnn_cell.LSTMCell(u)
        cell_bw = tf.nn.rnn_cell.LSTMCell(u)
        (left_output_fw_seq, left_output_bw_seq), left_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                                left_embeddings,
                                                                                                dtype=tf.float32)
        left_result = tf.concat([left_output_fw_seq,left_output_bw_seq],axis=-1)
    with tf.variable_scope("right_one_layer_bi-lstm"):
        (right_output_fw_seq, right_output_bw_seq), right_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                                   right_embeddings,
                                                                                                   dtype=tf.float32)
        right_result = tf.concat([right_output_fw_seq,right_output_bw_seq],axis=-1)
    with tf.variable_scope("attention"):
        attention_weights = tf.matmul(left_result,tf.transpose(right_result,[0,2,1]))
        attentionsoft_a = tf.nn.softmax(attention_weights)
        attentionsoft_b = tf.nn.softmax(tf.transpose(attention_weights,[0,2,1]))
        # attentionsoft_a [batch_size,max_time,max_time]
        # right_result [batch_size,max_time,hidden_size]
        left_hat = tf.matmul(attentionsoft_a, right_result)
        right_hat = tf.matmul(attentionsoft_b, left_result)
    with tf.variable_scope("compute"):
        left_diff = tf.subtract(left_result, left_hat)
        left_mul = tf.multiply(left_result, left_hat)

        right_diff = tf.subtract(right_result, right_hat)
        right_mul = tf.multiply(right_result, right_hat)

        m_left = tf.concat([left_result,left_hat,left_diff,left_mul],axis=2)
        m_right = tf.concat([right_result,right_hat,right_diff,right_mul],axis=2)

    with tf.variable_scope("bi-lstm"):
        cell_fw = tf.nn.rnn_cell.LSTMCell(d)
        cell_bw = tf.nn.rnn_cell.LSTMCell(d)

        (left_output_fw, left_output_bw), left_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,m_left,dtype=tf.float32)
        left_output = tf.concat([left_output_fw, left_output_bw], axis=-1)
        v_left_avg = tf.reduce_mean(left_output, axis=1)
        v_left_max = tf.reduce_max(left_output, axis=1)

        (right_output_fw, right_output_bw), right_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,m_right,dtype=tf.float32)
        right_output = tf.concat([right_output_fw, right_output_bw], axis=-1)
        v_right_avg = tf.reduce_mean(right_output, axis=1)
        v_right_max = tf.reduce_max(right_output, axis=1)

        # v_left = tf.concat([v_left_avg, v_left_max], axis=1)
        # v_right = tf.concat([v_right_avg, v_right_max], axis=1)
        v_concat = tf.concat([v_left_avg,v_left_max,v_right_avg,v_right_max],axis=1)

    with tf.variable_scope("classification"):
        # logits:shape[batch_size,num_tags]
        # output = tf.layers.dense(inputs=v_concat,units=32,activation=tf.nn.relu)
        output = tf.layers.dense(inputs=v_concat,units=8,activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=output, units=2)

    # 计算损失
    with tf.variable_scope("loss"):
        logits__ = tf.nn.softmax(logits)
        loss = (-0.25 * tf.reduce_sum(labels[:, 0] * tf.log(logits__[:, 0]))
                - 0.75 * tf.reduce_sum(labels[:, 1] * tf.log(logits__[:, 1]))
                ) / tf.cast(tf.shape(labels)[0],tf.float32)
        # losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        # loss = tf.reduce_mean(losses)

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
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))

        epsilon = 1e-7
        cm = tf.contrib.metrics.confusion_matrix(true, pred, num_classes=2)
        precision = cm[1][1] / tf.reduce_sum(tf.transpose(cm)[1])
        recall = cm[1][1] / tf.reduce_sum(cm[1])
        fbeta_score = (2 * precision * recall / (precision + recall + epsilon))



with tf.Session(graph=graph_mhd) as sess:
    if params.isTrain:
        saver = tf.train.Saver(tf.global_variables())
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(params.epoch_num):
            for s1, s2, y in data_utils.get_batch(train_data, params.batch_size, shuffle=True):
                _, l, acc, p, r, f,global_nums, cmm = sess.run(
                    [train_op, loss,accuracy,precision,recall,fbeta_score,global_add, cm], {
                    left_input: s1,
                    right_input: s2,
                    labels : y,
                    dropout_pl: 0.5
                })
                if global_nums % 50 == 0:
                    print(cmm)
                    saver.save(sess, '../model_save/model.ckpt', global_step=global_nums)
                    print(
                        'train: epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4} , precision {}, recall: {:.4}, fbeta_score: {:.4} '.format(epoch , global_nums,
                                                                                          l, acc,p, r, f))


                if global_nums % 200 == 0:
                    print('-----------------valudation---------------')
                    s1, s2, y = next(data_utils.get_batch(test_data, np.shape(test_data)[0],  shuffle=True))
                    l, acc, p, r, f, global_nums, cmm = sess.run(
                        [ loss, accuracy, precision, recall, fbeta_score, global_add, cm], {
                            left_input: s1,
                            right_input: s2,
                            labels: y,
                            dropout_pl: 1.
                        })
                    print(cmm)
                    print(
                        'valudation: epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4} , precision {}, recall: {:.4}, fbeta_score: {:.4} '.format(
                            epoch, global_nums,
                            l, acc, p, r, f))
                    print('-----------------train---------------')
        s1, s2, y = next(data_utils.get_batch(test_data, np.shape(test_data)[0], shuffle=True))
        l, acc, p, r, f, global_nums, cmm = sess.run(
                    [loss, accuracy, precision, recall, fbeta_score, global_add, cm], {
                        left_input: s1,
                        right_input: s2,
                        labels: y,
                        dropout_pl: 1.
                    })
        print(cmm)
        print('test: loss: {:.4}, accuracy: {:.4} , precision {}, recall: {:.4}, fbeta_score: {:.4} '.format(
                        l, acc, p, r, f))




























