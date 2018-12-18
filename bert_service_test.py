#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
from utils import *
import tensorflow as tf
from data_load import get_batch_data,load_vocab
from Config import BilstmCrfConfig as config
from bert_serving.client import BertClient
bc = BertClient()

sentence = [["test","dog"],["name","is"]]
vec = bc.encode(sentence,is_tokenized=True)

print(vec[0])
print(vec[0][1])

