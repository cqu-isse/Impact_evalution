#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
by: FuYing
date:2020.11.10
function: 将模版转为词向量
next step: 添加词干化等数据预处理步骤
存在问题：程序抛出警告：Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
'''

import os
import random

import numpy as np
import fasttext
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import json
from datetime import datetime
import mlflow
from mlflow import log_param, log_metrics, log_artifacts

model_dir = '/home/fuying/data_hub/model_hub/vec_model/wiki.en.text.ftmodel_300.bin'
model = fasttext.load_model(model_dir)

def events_to_wordcount(in_file):
    event_dataframe = pd.read_csv(in_file,header="infer")
    print(len(event_dataframe))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(list(event_dataframe['EventTemplate']))
    words = vectorizer.get_feature_names()
    return X.toarray(), words



def get_word_weight_by_tfidf(wordcount):
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(wordcount)
    return tfidf.toarray()


def event_to_vec(in_file,out_file):
    wordcount, words = events_to_wordcount(in_file)
    weights = get_word_weight_by_tfidf(wordcount)
    event_vec_dic = {}
    for i in range(len(weights)):
        vec_sum = 0
        count = 0
        for j in range(len(words)):
            if weights[i][j] > 0:
                word_vec = model.get_word_vector(words[j])
                vec_sum += word_vec*weights[i][j]
                count += 1
        try:
            event_vec = vec_sum/count
        #捕捉模版中只有特殊字符所导致的异常（没有得到词向量）
        except:
            event_vec = np.zeros(300)
            event_vec[2] = i*0.00001
        event_vec_dic[str(i)] = event_vec.tolist()
    json.dump(event_vec_dic,open(out_file,'w'))


if __name__ == "__main__":
    parser = 'Spell'
    dataset = 'smallTB'
    in_file = '/nas/fuying/Data_for_time_test/BGL/ground_truth/BGL.log_templates.csv'
    out_file = '/nas/fuying/Data_for_time_test/BGL/ground_truth/train_test_data_4_robust/event2semantic_vec.json'
    starttime = datetime.now()
    event_to_vec(in_file,out_file)



