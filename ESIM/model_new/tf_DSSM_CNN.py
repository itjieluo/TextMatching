import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')
from data_utils import data_utils
import params
import pickle

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

data_pk = '../data_utils/word_char_data.pk'
with open(data_pk, 'rb') as f:
    train_set,test_set,word2id,id2word,char2id,id2char = pickle.load(f)

word_embeddings = data_utils.random_embedding(id2word, params.embedding_dim)
char_embeddings = data_utils.random_embedding(id2char, params.embedding_dim)

graph = tf.Graph()
with graph.as_default():
    """
    构建神经网络的结构、损失、优化方法和评估方法
    """
    # shape[batch_size, max_sentence_length]
    left_input = tf.placeholder(tf.int32, shape=[None, params.max_sentence_length], name="left_input")
    # shape[batch_size, max_sentence_length]
    right_input = tf.placeholder(tf.int32, shape=[None, params.max_sentence_length], name="right_input")
    # shape[batch_size, max_word_length]
    left_c_input = tf.placeholder(tf.int32, shape=[None, params.max_word_length], name="left_c_input")
    # shape[batch_size, max_word_length]
    right_c_input = tf.placeholder(tf.int32, shape=[None, params.max_word_length], name="right_c_input")

    # shape[batch_size, sentences, labels]
    labels = tf.placeholder(tf.float32, shape=[None, 2], name="labels")
    
    with tf.variable_scope("embeddings"):#命名空间
        _word_embeddings = tf.Variable(word_embeddings,
                                       dtype=tf.float32,
                                       trainable=True,
                                       name="embedding_matrix")
        left_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,ids=left_input,name="left_embeddings")
        right_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,ids=right_input,name="right_embeddings")
    
        _char_embeddings = tf.Variable(char_embeddings,#shape[len_words,300]
                                       dtype=tf.float32,
                                       trainable=True,#嵌入层是否可以训练
                                       name="embedding_char_matrix")
        left_c_embeddings = tf.nn.embedding_lookup(params=_char_embeddings,ids=left_c_input,name="left_c_embeddings")
        right_c_embeddings = tf.nn.embedding_lookup(params=_char_embeddings,ids=right_c_input,name="right_c_embeddings")

    with tf.variable_scope("Attention"):#命名空间
        #在词Embeding层上添加attention层
        attention_w = tf.get_variable('attention_omega', [params.embedding_dim, 1])
        attention_b = tf.get_variable('attention_b', [1,params.max_sentence_length])
        attention_left = tf.reduce_sum(left_embeddings * tf.expand_dims(tf.nn.softmax(tf.tanh(tf.add(
            tf.reshape(tf.matmul(tf.reshape(left_embeddings, [-1, params.embedding_dim]), attention_w),[-1, params.max_sentence_length]),
            attention_b))), -1),axis = 1)
        attention_right = tf.reduce_sum(right_embeddings * tf.expand_dims(tf.nn.softmax(tf.tanh(tf.add(
            tf.reshape(tf.matmul(tf.reshape(right_embeddings, [-1, params.embedding_dim]), attention_w),[-1, params.max_sentence_length]),
            attention_b))), -1),axis = 1)
        
        #在字Embeding层上添加attention层
        attention_c_w = tf.get_variable('attention_c_omega', [params.embedding_dim, 1])
        attention_c_b = tf.get_variable('attention_c_b', [1,params.max_word_length])
        attention_c_left = tf.reduce_sum(left_c_embeddings * tf.expand_dims(tf.nn.softmax(tf.tanh(tf.add(
            tf.reshape(tf.matmul(tf.reshape(left_c_embeddings, [-1, params.embedding_dim]), attention_c_w),[-1, params.max_word_length]),
            attention_c_b))), -1),axis = 1)
        attention_c_right = tf.reduce_sum(right_c_embeddings * tf.expand_dims(tf.nn.softmax(tf.tanh(tf.add(
            tf.reshape(tf.matmul(tf.reshape(right_c_embeddings, [-1, params.embedding_dim]), attention_c_w),[-1, params.max_word_length]),
            attention_c_b))), -1),axis = 1)
        
    with tf.variable_scope("one_layer_word_bi-lstm"):
        #词1层bi-lstm
        cell_fw = tf.nn.rnn_cell.LSTMCell(50)
        cell_bw = tf.nn.rnn_cell.LSTMCell(50)
        (left_output_fw_seq, left_output_bw_seq), left_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
            left_embeddings,dtype=tf.float32)
        left_bi_output = tf.concat([left_states[0].h, left_states[1].h], axis=-1)

        (right_output_fw_seq, right_output_bw_seq), right_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
            right_embeddings,dtype=tf.float32)
        right_bi_output = tf.concat([right_states[0].h, right_states[1].h], axis=-1)
        
    with tf.variable_scope("one_layer_char_bi-lstm"):
        #字1层bi-lstm
        cell_c_fw = tf.nn.rnn_cell.LSTMCell(6)
        cell_c_bw = tf.nn.rnn_cell.LSTMCell(6)
        (left_c_output_fw_seq, left_c_output_bw_seq), left_states = tf.nn.bidirectional_dynamic_rnn(cell_c_fw, cell_c_bw,
            left_c_embeddings,dtype=tf.float32)
        left_c_bi_output = tf.concat([left_states[0].h, left_states[1].h], axis=-1)

        (right_c_output_fw_seq, right_c_output_bw_seq), right_states = tf.nn.bidirectional_dynamic_rnn(cell_c_fw, cell_c_bw,
            right_c_embeddings,dtype=tf.float32)
        right_c_bi_output = tf.concat([right_states[0].h, right_states[1].h], axis=-1)
    
    with tf.variable_scope("two_layer_lstm"):
        #词2层单向lstm
        cell_1 = tf.nn.rnn_cell.LSTMCell(50)
        cell_2 = tf.nn.rnn_cell.LSTMCell(50)
        output,state=tf.nn.dynamic_rnn(cell=cell_1,inputs=left_embeddings,dtype=tf.float32)
        #第一层的output输出
        left_output = output

        lstm_cell = tf.nn.rnn_cell.MultiRNNCell([cell_1,cell_2])
        output,state=tf.nn.dynamic_rnn(cell=lstm_cell,inputs=left_embeddings,dtype=tf.float32)
        #第二层的h输出
        left_2_output = state[1].h
        
        output,state=tf.nn.dynamic_rnn(cell=cell_1,inputs=right_embeddings,dtype=tf.float32)
        #第一层的output输出
        right_output = output
        
        output,state=tf.nn.dynamic_rnn(cell=lstm_cell,inputs=right_embeddings,dtype=tf.float32)
        #第二层的h输出
        right_2_output = state[1].h 
        
    with tf.variable_scope("cnn"):
        #定义权值
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, mean=0,stddev=0.1)
            return tf.Variable(initial)
        #定义偏置
        def bias_variable(shape):
            initial = tf.constant(0.1,shape=shape)
            return tf.Variable(initial)
        #定义卷积
        def conv2d(x,W):
            #srtide[1,x_movement,y_,movement,1]步长参数说明
            return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding='SAME')
            #x为输入，W为卷积参数，[5,5,1,32]表示5*5的卷积核，1个channel，32个卷积核。strides表示模板移动步长，
            #SAME和VALID两种形式的padding，valid抽取出来的是在原始图片直接抽取，结果比原始图像小，
            #same为原始图像补零后抽取，结果与原始图像大小相同。
        #定义pooling
        def max_pool_2x2(x):
            #ksize   strides
            return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        cnn_input = tf.matmul(left_output, tf.transpose(right_output,[0,2,1]))#矩阵乘法
        cnn_input = tf.reshape(cnn_input,[-1,20,20,1])
        ##卷积层conv1
        W_conv1 = weight_variable([3,3,1,8])#第一层卷积：卷积核大小3x3,1个颜色通道，8个卷积核
        b_conv1 = bias_variable([8])#第一层偏置
        h_conv1 = tf.nn.relu(conv2d(cnn_input,W_conv1)+b_conv1)#第一层输出：输出的非线性处理20x20x8
        h_pool1 = max_pool_2x2(h_conv1)#输出为10x10x8
        ##卷积层conv2
        W_conv2 = weight_variable([3,3,8,6])#第二层卷积：卷积核大小3x3,1个颜色通道，6个卷积核
        b_conv2 = bias_variable([6])
        h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)#输出为5x5x6

        ##全连接层
        h_pool2_flat = tf.reshape(h_pool2,[-1,5*5*6])
        h_fc1 = tf.nn.dropout(h_pool2_flat, 0.9)
        h_fc2 = tf.layers.dense(h_fc1, 32,activation=tf.nn.relu)
        h_fc3 = tf.layers.dense(h_fc2, 16,activation=tf.nn.relu)

    with tf.variable_scope("concatenate_layer"):
        #词单层bi-lstm和attention拼接
        s1_last = tf.concat([attention_left, left_bi_output],1) 
        s2_last = tf.concat([attention_right, right_bi_output],1) 
        #字单层bi-lstm和attention拼接
        s1_c_last = tf.concat([attention_c_left, left_c_bi_output],1) 
        s2_c_last = tf.concat([attention_c_right, right_c_bi_output],1)
        
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

        #词相似度
        cos = cosine_dist(s1_last,s2_last)
        man = manhattan_dist(s1_last,s2_last)
        mul = multiply(s1_last,s2_last)
        sub = subtract(s1_last,s2_last)
        sub1 = subtract(left_2_output,right_2_output)

        #字相似度
        cos_c = cosine_dist(s1_c_last,s2_c_last)
        man_c = manhattan_dist(s1_c_last,s2_c_last)
        mul_c = multiply(s1_c_last,s2_c_last)
        sub_c = subtract(s1_c_last,s2_c_last)

    with tf.variable_scope("dense_layer"):
        last_list_layer = tf.concat([mul, sub, sub1, mul_c, sub_c],1)
        #last_list_layer = tf.concat([mul, sub, sub1, maxium ],1) 
        last_drop = tf.nn.dropout(last_list_layer, 0.8)
        dense_layer1 = tf.layers.dense(last_drop, 16,activation=tf.nn.relu)
        dense_layer2 = tf.layers.dense(last_drop, 24,activation=tf.nn.sigmoid)
        output = tf.concat([dense_layer1, dense_layer2, tf.expand_dims(cos,-1), tf.expand_dims(man,-1),
		tf.expand_dims(cos_c,-1), tf.expand_dims(man_c,-1),h_fc3],1)
        #output = tf.concat([dense_layer1, dense_layer2, tf.expand_dims(cos,-1), tf.expand_dims(man,-1)],1)

    with tf.variable_scope("classification"):
        #logits = tf.layers.dense(output, 2, activation=tf.nn.sigmoid)
        logits = tf.layers.dense(output, 2)

    #计算损失
    with tf.variable_scope("loss"):
        #losses = tf.nn.weighted_cross_entropy_with_logits(logits=labels_test,targets=labels,pos_weight = 5.0)
        logits__ = tf.nn.softmax(logits)
        losses = (-0.25 * tf.reduce_sum(labels[:, 0] * tf.log(logits__[:, 0]))
                - 0.75 * tf.reduce_sum(labels[:, 1] * tf.log(logits__[:, 1]))
                ) / params.batch_size
        # losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels)
        loss = tf.reduce_mean(losses)
        
    #选择优化器
    with tf.variable_scope("train_step"):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        global_add = global_step.assign_add(1)#用于计数
        train_op = tf.train.AdamOptimizer(params.lr).minimize(loss)
        
    #准确率/f1/p/r计算
    with tf.variable_scope("evaluation"):
        true = tf.cast(tf.argmax(labels, axis=-1), tf.float32)#真实序列的值
        labels_softmax = tf.nn.softmax(logits)
        labels_softmax_ = tf.argmax(labels_softmax, axis=-1)
        pred = tf.cast(labels_softmax_, tf.float32)#预测序列的值
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))

        epsilon = 1e-7
        cm = tf.contrib.metrics.confusion_matrix(true, pred, num_classes=2)
        precision = cm[1][1]/tf.reduce_sum(tf.transpose(cm)[1])
        recall = cm[1][1]/tf.reduce_sum(cm[1])
        fbeta_score = (2*precision*recall/(precision+recall+epsilon))


with tf.Session(graph=graph) as sess:
    if params.isTrain:
        saver = tf.train.Saver(tf.global_variables())
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(params.epoch_num):
            for s1, s2, c1,c2, y in data_utils.get_batch_all(train_set, params.batch_size, shuffle=True):
                _, l, acc, p, r, f,global_nums, cmm = sess.run(
                    [train_op, loss,accuracy,precision,recall,fbeta_score,global_add,cm], {
                    left_input: s1,
                    right_input: s2,
                    left_c_input: c1,
                    right_c_input: c2,
                    labels : y,
                })
                if global_nums % 50 == 0:
                    print(cmm)
                    # saver.save(sess, './checkpoint/model.ckpt', global_step=global_nums)
                    print(
                        'train: epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4} , precision {}, recall: {:.4}, fbeta_score: {:.4} '.format(epoch , global_nums,
                                                                                          l, acc,p, r, f))

                    # target_names = ['0', '1']
                    # print(classification_report(t, p, target_names=target_names))
                if global_nums % 200 == 0:
                    print('-----------------valudation---------------')
                    s1, s2, c1, c2, y = next(data_utils.get_batch_all(test_set, 20000,  shuffle=True))
                    l, acc, p, r, f, global_nums, cmm = sess.run(
                        [ loss, accuracy, precision, recall, fbeta_score, global_add,cm], {
                            left_input: s1,
                            right_input: s2,
                            left_c_input: c1,
                            right_c_input: c2,
                            labels: y,
                        })
                    print(cmm)
                    print(
                        'valudation: epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4} , precision {}, recall: {:.4}, fbeta_score: {:.4} '.format(
                            epoch, global_nums,
                            l, acc, p, r, f))
                    # target_names = ['0', '1']
                    # print(classification_report(t, p, target_names=target_names))
                    print('-----------------valudation---------------')
        s1, s2, c1, c2, y = next(data_utils.get_batch_all(test_set, 20000, shuffle=True))
        l, acc, p, r, f, global_nums, cmm = sess.run(
            [loss, accuracy, precision, recall, fbeta_score, global_add, cm], {
                left_input: s1,
                right_input: s2,
                left_c_input: c1,
                right_c_input: c2,
                labels: y,
            })
        print(cmm)
        print('test: loss: {:.4}, accuracy: {:.4} , precision {}, recall: {:.4}, fbeta_score: {:.4} '.format(
            l, acc, p, r, f))
