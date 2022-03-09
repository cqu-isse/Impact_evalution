#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import DecisionTree
from loglizer import dataloader, preprocessing
from datetime import datetime
import os
Project_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

def test_run(parser, dataset):
    struct_log = os.path.join(Project_path, 'logparser/parsing_result/'+dataset+'/'+parser+'/HDFS.log_structured.csv')
    label_file = os.path.join(Project_path, 'logparser/parsing_result/'+dataset+'/'+parser+'/anomaly_label.csv') # The anomaly label file
    result_file = os.path.join(Project_path, 'ML_loglizer/detection_result/'+dataset+'/'+parser+'/DecisionTree_reuslt.csv')

    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = dataloader.load_HDFS(struct_log,
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.8,
                                                                split_type='uniform')
    round = 5
    for i in range(round):
        x_train = x_train_raw
        y_train = y_train_raw
        x_test = x_test_raw
        y_test = y_test_raw
        feature_extractor = preprocessing.FeatureExtractor()
        starttime = datetime.now()
        x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
        x_test = feature_extractor.transform(x_test)

        model = DecisionTree()
        model.fit(x_train, y_train)
        print('Train validation:')
        precision, recall, f1 = model.evaluate(x_train, y_train)

        print('Test validation:')
        precision, recall, f1 = model.evaluate(x_test, y_test)
        finish_time = datetime.now() - starttime
        print(finish_time)
        with open(result_file,'a+',encoding='utf-8') as f:
            f.write(','.join([str(precision), str(recall), str(f1), str(finish_time)])+'\n')

if __name__ == '__main__':
    log_parsers = ['Drain','IPLoM','Logram','LFA','Lenma','Spell','ground_truth']
    dataset = 'HDFS'
    for parser in log_parsers:
        test_run(parser,dataset)
