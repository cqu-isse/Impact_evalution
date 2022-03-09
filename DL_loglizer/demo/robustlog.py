#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import os
sys.path.append('../')

from logdeep.models.lstm import deeplog, loganomaly, robustlog, robustlog_s
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *
from datetime import datetime

Project_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

dataset = 'HDFS'
logparser = 'Logram'

num_templates_dic = {'HDFS_Drain':48,'HDFS_IPLoM':41,'HDFS_Spell':37, 'HDFS_Logram':97}

# Config Parameters
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
options = dict()

# options['data_dir'] = '/nas/fuying/data_hub/'+ dataset + '/train_test_data_4_robust/' + logparser+'_v2'

options['data_dir'] = os.path.join(Project_path, 'DL_loglizer/log_data/'+dataset+'/'+logparser+'/train_test_data_4_robust/')

# options['window_size'] = 10
# options['window_size'] = 4
options['device'] = "cuda"
options['num_templates'] = num_templates_dic[dataset+'_'+logparser]
# options['device'] = "cpu"

# Smaple
options['sample'] = "session_window"
# options['sample'] = "fixed_window"
options['window_size'] = -1

# Features
options['sequentials'] = False
options['quantitatives'] = False
options['semantics'] = True
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics']])

# Model
options['input_size'] = 300
# options['hidden_size'] = 128
options['hidden_size'] = 256
options['num_layers'] = 2
options['num_classes'] = 2

# Train
options['batch_size'] = 256
options['accumulation_step'] = 1

options['optimizer'] = 'adam'
options['lr'] = 0.001
options['max_epoch'] = 200
options['lr_step'] = (160, 180)
options['lr_decay_ratio'] = 0.01

options['resume_path'] = None
options['model_name'] = "robustlog"

options['save_dir'] = os.path.join(Project_path, 'DL_loglizer/model_hub/'+dataset+'/'+logparser+'/robustlog/')

# Predict
options['model_path'] = os.path.join(Project_path, 'DL_loglizer/model_hub/'+dataset+'/'+logparser+'/robustlog/robustlog_last.pth')
options['num_candidates'] = -1

# seed_everything(seed=23)


def train():
    Model = robustlog_s(input_size=options['input_size'],
                      hidden_size=options['hidden_size'],
                      num_layers=options['num_layers'],
                      num_keys=options['num_classes'])
    trainer = Trainer(Model, options)
    starttime = datetime.now()
    trainer.start_train()


def predict():
    Model = robustlog_s(input_size=options['input_size'],
                      hidden_size=options['hidden_size'],
                      num_layers=options['num_layers'],
                      num_keys=options['num_classes']
                      )
    predicter = Predicter(Model, options)
    TP, FP, FN, TN = predicter.predict_supervised()
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'])
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        predict()
