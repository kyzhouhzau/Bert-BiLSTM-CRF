#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
from utils import *
import tensorflow as tf
from data_load import get_batch_data,load_vocab
from Config import BilstmCrfConfig as config
class Model_Graph():
    def __init__(self,is_training):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.is_training = is_training

            if self.is_training:
                self.next_element,num_batch = get_batch_data(self.is_training)
                self.X, self.Y, self.seq_len = self.next_element["X"], self.next_element["Y"], self.next_element["seq_len"]
                self.X.set_shape([None, config.maxlen,config.bert_dim])
                self.Y.set_shape([None, config.maxlen])
                self.seq_len.set_shape([None])
            else:
                self.X = tf.placeholder(tf.float32, shape=(None, config.maxlen,config.bert_dim))
                self.Y = tf.placeholder(tf.int32, shape=(None, config.maxlen))
                self.seq_len = tf.placeholder(tf.int32, shape=(None))
            
            idx2word, word2idx, idx2labl, labl2idx  = load_vocab()
            embed = tf.convert_to_tensor(self.X)
            # input embedding Dropout
            embed = tf.layers.dropout(embed,rate=config.dropout_rate,training=self.is_training)
            #Muilty layer Bilstm 
            outputs = multibilstm(embed,self.seq_len,config.num_units,config.num_layer,self.is_training,config.cell)
            
            #full connect layer
            # here we use two layer full connect layer. residual and activation can be set by your self. 
            outputs = feedforward(outputs,outputs.get_shape().as_list()[2],scope="first")#residual default used
            outputs = feedforward(outputs,config.num_class,residual=False,scope="second")
            noutput = tf.reshape(outputs, [-1, config.maxlen, config.num_class])

            # crf layer
            if config.use_crf:
                self.loss, self.acc, self.predicts,self.true_labels = crf_layer(self.Y,noutput,config.num_class,self.seq_len,self.is_training)
            else:
                self.loss, self.acc, self.predicts, self.true_labels = loss_layer(self.Y, noutput, config.num_class)
            tf.summary.scalar('acc',self.acc)
            self.global_step = tf.Variable(0, name='global_step')
            if self.is_training:
                # use exponential_decay to help the model fit quicker
                if config.exponential_decay:
                    learning_rate = tf.train.exponential_decay(
                        config.lr,self.global_step, 200, 0.96, staircase=True
                    )
                # optimizer = tf.train.AdamOptimizer(learning_rate=config.lr, beta1=0.9, beta2=0.99, epsilon=1e-8)
                optimizer = tf.train.RMSPropOptimizer(learning_rate=config.lr)
                self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
                tf.summary.scalar('mean_loss',self.loss)
            else:
                train_op=None
