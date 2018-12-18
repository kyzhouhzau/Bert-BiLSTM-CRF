#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
from train import Model_Graph
from data_load import *
from utils import *
from tqdm import tqdm
def eval():
    model_graph = Model_Graph(False)
    
    X, Y, sentence_word, sentence_tag, seq_len = load_testdata()
    allpredicts = []
    ally = []
    with model_graph.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session() as sess:
            sv.saver.restore(sess,tf.train.latest_checkpoint('./logdir/model'))
            print("Model was loaded successfully!")
            #预测时候以单句话为单位预测
            for i in tqdm(range(len(X))):
                x = X[i:i+1]
                y=Y[i:i+1]
                s_l = seq_len[i:i+1]
                x,y,s_l = get_encodes(x,y,s_l)
                s_words = sentence_word[i:i+1]
                s_tags = sentence_tag[i:i+1]
                _predics = sess.run(model_graph.predicts,{model_graph.X:x,model_graph.Y:y,model_graph.seq_len:s_l})
                #通过实际长度来获取预测和实际标签
                slice_ = [int(i) for i in s_l]
                pred = np.array([_predics[:,:index] for index in slice_]).reshape(-1)
                y = np.array([y[:,:index] for index in slice_]).reshape(-1)

                allpredicts.extend(pred)
                ally.extend(y)
            eval_result(np.array(ally),np.array(allpredicts))

if __name__=="__main__":
    eval()



