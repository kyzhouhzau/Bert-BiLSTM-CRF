
#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""

import os
import sklearn
import gensim
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import initializers
from Config import BilstmCrfConfig as config
from sklearn.metrics import f1_score,recall_score,precision_score
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

def normalize(inputs, epslion=1e-8, scope="ln", reuse=None):
    """
    Here we use batch normalize to reduce variation
    :param inputs: out put of each layer
    :param epslion: smoothing para
    :param scope: scope name
    :param resuse: if None it will use fathers.
    :return: outputs
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epslion) ** 0.5)
        outputs = gamma * normalized + beta
        return outputs

def embedding(inputs,use_pretrain=False,scope="embedding",reuse=None):
    """
    embedding look up to get embedding for each batch!
    """
    with tf.variable_scope(scope,reuse=reuse):
        outputs = inputs
    return outputs

def position_encoding(inputs,position_dim,sinusoid=False,scale=False,reuse=None,scope="positional_encoding"):
    """
    position embedding to capture position features,True or False
    :param inputs: batch data int.
    :param position_dim:to capture position feature
    :param scale:true or false
    :param reuse:if None use father scope
    :return:outputs
    """
    N,L = inputs.get_shape().as_list()
    with tf.variable_scope(scope,reuse=reuse):
        position_id = tf.tile(tf.expand_dims(tf.range(L), 0), [N, 1])#[N,L,1]
        if sinusoid:
            position_enc = np.array([[pos/np.power(10000,2*i/position_dim) for i in range(position_dim)]
                                     for pos in range(L)])
            position_enc[:,0::2] = np.sin(position_enc[0,0::2])
            position_enc[:,1::2] = np.cos(position_enc[0,1::2])
            lookup_table = tf.convert_to_tensor(position_enc)
            lookup_table = tf.concat((tf.zeros(shape=[1,position_dim]),lookup_table[1:,:]),0)
        else:
            position_enc = np.array([[np.random.normal(0,0.1,[position_dim])] for _ in range(L)])
            lookup_table = tf.convert_to_tensor(position_enc)
            lookup_table = tf.concat((tf.zeros(shape=[1,position_dim]),lookup_table[1:,:]),0)
        outputs = tf.nn.embedding_lookup(lookup_table,position_id)
        if scale:
            outputs = outputs/position_dim**0.5
        return outputs

def feedforward(inputs,num_units,reuse=None,conv=False,residual=False,scope=None):
    #Use full connection layer you can chose conv method or usual method!
    with tf.variable_scope(scope,reuse=reuse):
        if conv:
            params ={
                "inputs":inputs,
                "filters":num_units,
                "kernel_size":1,
                "activation":tf.nn.relu,
                "use_bias":True
            }
            outputs = tf.layers.conv1d(**params)
        else:
            params = {
                "inputs": inputs,
                "units": num_units,
                "kernel_initializer":initializers.xavier_initializer(),
                "activation": tf.nn.relu,
                "use_bias": True
            }
            if config.regulation:
                #输出的正则化
                params["activity_regularizer"] = tf.contrib.layers.l2_regularizer(0.001)
            outputs = tf.layers.dense(**params)
        if residual:
            outputs+=inputs
        if config.normalize:
            outputs = normalize(outputs)
        return outputs

def _cell(cell,num_units):
    if cell == "lstm":
        cell = rnn.LSTMCell(num_units)
    elif cell == "gru":
        cell = rnn.GRUCell(num_units)
    else:
        print("VALUE ERROR WARNING:Please set para cell='lstm' or cell='gru'!")
    return cell

def multibilstm(inputs,seq_len,num_units,num_layer,is_training=True,cell="lstm",keeprate=0.8,scope="Multi_RNN"):
    with tf.variable_scope(scope):
        for i in range(num_layer):
            with tf.variable_scope("Bilstmlyer" + str(i)):
                cell_fw = _cell(cell,num_units)
                cell_bw = _cell(cell,num_units)
                if is_training:
                    cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=keeprate)
                    cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=keeprate)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                             cell_bw,
                                                             inputs,
                                                             sequence_length=seq_len,
                                                             dtype=tf.float32)
                inputs = tf.concat(outputs, axis=2)
        return inputs

def crf_layer(labels,logits,num_class,seq_len,is_training):
    #add crf_layer
    with tf.variable_scope("crf_loss"):
        trans = tf.get_variable(
            "transition",
            shape=[num_class,num_class],
            initializer=initializers.xavier_initializer()
        )
        log_likelihood,trans = tf.contrib.crf.crf_log_likelihood(
            inputs=logits,
            tag_indices=labels,
            transition_params=trans,
            sequence_lengths=seq_len
        )
        # if is_training:
        decode_tags, best_score = tf.contrib.crf.crf_decode(logits,trans, seq_len)
        # else:
            # decode_tags, best_score =  tf.contrib.crf.viterbi_decode(logits,trans)

        istarget = tf.to_float(tf.not_equal(labels,0))
        acc = tf.reduce_sum(tf.to_float(tf.equal(decode_tags,labels))*istarget)/(tf.reduce_sum(istarget))
        npredicts = tf.to_float(decode_tags)*istarget
        true_labels= tf.convert_to_tensor(labels)
        loss = tf.reduce_mean(-log_likelihood)
        return loss,acc,npredicts,true_labels
        
def loss_layer(labels,logits,num_class):
    true_labels = labels
    predicts = tf.to_int32(tf.argmax(logits, dimension=-1))
    istarget = tf.to_float(tf.not_equal(labels, 0))
    npredicts = tf.to_float(predicts)*istarget
    acc = tf.reduce_sum(tf.to_float(tf.equal(predicts, labels)) * istarget) / (tf.reduce_sum(istarget))
    y_smmothed = label_smoothing(tf.one_hot(labels, depth=num_class))
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_smmothed)
    loss = tf.reduce_sum(loss * istarget) / (tf.reduce_sum(istarget))
    return loss,acc,npredicts,true_labels

def label_smoothing(inputs,epsilon=0.1):
    """
    label smooth method if you don't know google for it
    :param inputs:
    :param epsilon:
    :return:
    """
    K = inputs.get_shape().as_list()[-1]
    outputs = ((1-epsilon)*inputs)+(epsilon/K)
    return outputs

def embedding_trim(idx2word,trimed_path):
    #if you change your data please remove trimed embedding, otherwise you will get wrong results!
    if not os.path.exists(trimed_path):
        trimed = []
        model = gensim.models.KeyedVectors.load_word2vec_format(config.embedding_path)
        glove_vocabs = model.vocab.keys()
        count=0
        for idx in range(len(idx2word)):
            word = idx2word[idx]
            if word in glove_vocabs:
                trimed.append(model[word])
            else:
                count+=1
                ebed = np.random.normal(0,0.1,[config.embed_dim])
                trimed.append(ebed)
        embed = np.array(trimed).astype(np.float32)
        print("When we trimed embedding we find about {} words cant's embedding can't be fined in "
              "current embeddings!".format(count))
        np.save(trimed_path,embed)

def eval_result(true_labels,predicts,acc=0.0,loss=0.0,epoch=0):
    true_labels= true_labels.reshape(-1)
    predicts= predicts.reshape(-1)
    precision = precision_score(true_labels, predicts,labels=list(range(3,12)), average="macro")
    recall = recall_score(true_labels, predicts,labels=list(range(3,12)), average="macro")
    fscore = f1_score(true_labels, predicts,labels=list(range(3,12)), average="macro")
    print("EPOCH:{}\tF:{:.3f}\tP:{:.3f}\tR:{:.3f}\tACC:{:.3f}\tLOSS:{:.3f}".format(epoch,fscore, precision, recall, acc, loss))

