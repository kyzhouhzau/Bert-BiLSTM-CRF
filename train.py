#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""

from Bert_Bilstm_crf import Model_Graph
from utils import eval_result
from data_load import get_batch_data,load_vocab
from Config import BilstmCrfConfig as config
import tensorflow as tf
def train():
    model_graph = Model_Graph(True)
    sv = tf.train.Supervisor(
        graph=model_graph.graph,
        logdir=config.logdir,
        save_model_secs=100,
        checkpoint_basename='./model/model.ckpt',
        global_step=model_graph.global_step,
        summary_writer=tf.summary.FileWriter(r'./logdir/summary/')
    )
    with sv.managed_session() as sess:
        counter=0
        try:
            while True:
                if sv.should_stop(): break
                counter+=1
                sess.run(model_graph.train_op)
                true_label, predict, accs, losses = sess.run([model_graph.true_labels,model_graph.predicts,model_graph.acc,model_graph.loss])
                eval_result(true_label,predict,accs,losses,counter)
        except tf.errors.OutOfRangeError:
            print("Train finished")

if __name__=="__main__":
    train()














