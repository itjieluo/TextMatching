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
embeddings = data_utils.random_embedding(id2word, params.embedding_dim)
# fname = '../data_utils/word2vec_model.pkl'
# embeddings = data_utils.word2vec_embedding_big_corpus(word2id, params.embedding_dim, fname)


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
    labels_ = tf.reshape(tf.cast(tf.argmax(labels, axis=-1), tf.float32), (-1, 1))

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
    with tf.variable_scope("seq_mix"):
        all_feature_map = []
        for i in range(params.batch_size):
            one_feature_map = []
            left_input_one = left_embeddings[i]
            right_input_one = right_embeddings[i]
            for j in range(params.max_sentence_length - 2):
                left_rows = left_input_one[j:j + 3]
                for k in range(params.max_sentence_length - 2):
                    right_rows = right_input_one[k:k + 3]
                    one_dot_feature = tf.concat((left_rows, right_rows), axis=0)
                    for n in range(6):
                        one_feature_map.append(one_dot_feature[n])
            all_feature_map.append(one_feature_map)
        all_feature_maps = tf.cast(all_feature_map, tf.float32)

    with tf.variable_scope("conv1d_"):
        conv_1d = tf.layers.conv1d(inputs=all_feature_maps,
                                                  filters=200,
                                                  kernel_size=6,
                                                  strides=6,
                                                  padding="same",
                                                  activation=tf.nn.sigmoid,
                                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        mix_figure = tf.reshape(conv_1d, (params.batch_size, params.max_sentence_length - 2, params.max_sentence_length - 2, 200))
        mix_figure = tf.layers.max_pooling2d(mix_figure, pool_size=[2, 2], strides=2, padding="same")

    with tf.variable_scope('conv2d_'):
        conv2d_result_1 = tf.layers.conv2d(mix_figure,
                                         filters=100,
                                         kernel_size=[3, 3],
                                         padding="same",
                                         activation=tf.nn.relu,
                                         strides=[1, 1],
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        pool2d_result_1 = tf.layers.max_pooling2d(conv2d_result_1, pool_size=[2, 2], strides=2, padding="same")
        conv2d_result_2 = tf.layers.conv2d(pool2d_result_1,
                                         filters=100,
                                         kernel_size=[3, 3],
                                         padding="same",
                                         activation=tf.nn.relu,
                                         strides=[1, 1],
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        pool2d_result_2 = tf.layers.max_pooling2d(conv2d_result_2, pool_size=[2, 2], strides=2, padding="same")
        output = tf.reshape(pool2d_result_2, (params.batch_size, -1))


    with tf.variable_scope("classification"):
        # logits:shape[batch_size, labels]
        output = tf.layers.dense(output, 256)
        output = tf.layers.dense(output, 32)
        logits = tf.layers.dense(output, 1)

    # 计算损失
    with tf.variable_scope("loss"):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_, logits=logits))

        # 选择优化器
        with tf.variable_scope("train_step"):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            global_add = global_step.assign_add(1)  # 用于计数
            train_op = tf.train.AdamOptimizer(params.lr).minimize(loss)

        # 准确率/f1/p/r计算
        with tf.variable_scope("evaluation"):
            pred = tf.cast(tf.greater(tf.sigmoid(logits), 0.26), tf.float32)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, labels_), tf.float32), name="accuracy")
            # 混淆矩阵
            # _|0 |1 |
            # 0|2 |3 |
            # 1|2 |3 |

            true = tf.reshape(labels_, (-1,))
            pred = tf.reshape(pred, (-1,))

            epsilon = 1e-7
            cm = tf.contrib.metrics.confusion_matrix(true, pred, num_classes=2)

            precision = tf.cast(cm[1][1] / tf.reduce_sum(cm[:, 1]), tf.float32, name="precision")
            recall = tf.cast(cm[1][1] / tf.reduce_sum(cm[1], axis=0), tf.float32, name="recall")
            f1_score = tf.cast((2 * precision * recall / (precision + recall + epsilon)), tf.float32, name="f1_score")

    with tf.Session(graph=graph_mhd) as sess:
        if params.isTrain:
            saver = tf.train.Saver(tf.global_variables())
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(params.epoch_num):
                for s1, s2, y in data_utils.get_batch(train_set, params.batch_size, shuffle=True):
                    _, l, acc, p, r, f, global_nums, cmm = sess.run(
                        [train_op, loss, accuracy, precision, recall, f1_score, global_add, cm], {
                            left_input: s1,
                            right_input: s2,
                            labels: y,
                            dropout_pl: params.dropout
                        })

                    if global_nums % 50 == 0:
                        print(cmm)
                        # saver.save(sess, '../model_save/model.ckpt', global_step=global_nums)
                        print(
                            'train: epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4} , precision {}, recall: {:.4}, fbeta_score: {:.4} '.format(
                                epoch, global_nums,
                                l, acc, p, r, f))

                    if global_nums % 200 == 0:
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
