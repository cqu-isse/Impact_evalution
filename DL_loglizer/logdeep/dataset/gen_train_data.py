#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
by: FuYing
date:2020.11.10
'''

import os
import random
# from random import shuffle

import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import mlflow
from mlflow import log_param
from mlflow import log_artifact

# mlflow.set_tracking_uri("http://127.0.0.1:8080")
# mlflow.set_experiment("train_test_data split")

def data_read(filepath):
    fp = open(filepath, "r")
    datas = []  # 存储处理后的数据
    lines = fp.readlines()  # 读取整个文件数据
    i = 0  # 为一行数据
    count = 0
    for line in lines:
        row = line.strip()
        cons = row.split(' ')
        if len(cons)>9:
            datas.append(row)
            count += 1
        i = i + 1
    print('total, left======>',(i,count))
    fp.close()
    return datas

def train_test_gen(normal_in_file, abnormal_in_file, train_ratio, output_root_path):
    normal_datas = data_read(normal_in_file)
    abnormal_datas = data_read(abnormal_in_file)
    train_num = int(train_ratio*len(abnormal_datas))
    valid_num = train_num + int(0.05*len(abnormal_datas))
    train_file = output_root_path +'robust_log_train.csv'
    valid_file = output_root_path +'robust_log_valid.csv'
    test_file = output_root_path +'robust_log_test.csv'
    normal_datas = shuffle(normal_datas)
    abnormal_datas = shuffle(abnormal_datas)
    train_datas = []
    valid_datas = []
    test_datas = []
    train_labels = []
    valid_labels = []
    test_labels = []
    train_df = pd.DataFrame(columns=['Sequence','label'])
    valid_df = pd.DataFrame(columns=['Sequence','label'])
    test_df = pd.DataFrame(columns=['Sequence','label'])
    for i in range(len(abnormal_datas)):
        if i < train_num:
            train_datas.append(abnormal_datas[i])
            train_labels.append(1)
        elif (i >= train_num) & (i < valid_num):
            valid_datas.append(abnormal_datas[i])
            valid_labels.append(1)
        else:
            test_datas.append(abnormal_datas[i])
            test_labels.append(1)
    
    for i in range(len(normal_datas)):
        if i < train_num:
            train_datas.append(normal_datas[i])
            train_labels.append(0)
        elif (i >= train_num) & (i < valid_num):
            valid_datas.append(normal_datas[i])
            valid_labels.append(0)
        else:
            test_datas.append(normal_datas[i])
            test_labels.append(0)

    train_df['Sequence'] = train_datas
    train_df['label'] = train_labels
    valid_df['Sequence'] = valid_datas
    valid_df['label'] = valid_labels
    test_df['Sequence'] = test_datas
    test_df['label'] = test_labels
    # log_param('length of train_df',len(train_df))
    # log_param('length of valid_df',len(valid_df))
    # log_param('length of test_df',len(test_df))
    train_df.to_csv(train_file,index=0)
    valid_df.to_csv(valid_file,index=0)
    test_df.to_csv(test_file,index=0)

def train_test_gen_4_hdfs(normal_in_file, abnormal_in_file, train_ratio, output_root_path):
    normal_datas = data_read(normal_in_file)
    abnormal_datas = data_read(abnormal_in_file)
    train_num = 8000
    valid_num = 9000
    train_file = output_root_path +'robust_log_train.csv'
    valid_file = output_root_path +'robust_log_valid.csv'
    test_file = output_root_path +'robust_log_test.csv'
    # normal_datas = shuffle(normal_datas)
    # abnormal_datas = shuffle(abnormal_datas)
    train_datas = []
    valid_datas = []
    test_datas = []
    train_labels = []
    valid_labels = []
    test_labels = []
    train_df = pd.DataFrame(columns=['Sequence','label'])
    valid_df = pd.DataFrame(columns=['Sequence','label'])
    test_df = pd.DataFrame(columns=['Sequence','label'])
    total_len = (len(abnormal_datas) - valid_num)*50+valid_num
    for i in range(len(abnormal_datas)):
        if i < train_num:
            train_datas.append(abnormal_datas[i])
            train_labels.append(1)
        elif (i >= train_num) & (i < valid_num):
            valid_datas.append(abnormal_datas[i])
            valid_labels.append(1)
        else:
            test_datas.append(abnormal_datas[i])
            test_labels.append(1)
    
    for i in range(len(normal_datas)):
        total_len = (len(abnormal_datas) - valid_num)*50+valid_num
        if i < train_num:
            train_datas.append(normal_datas[i])
            train_labels.append(0)
        elif (i >= train_num) & (i < valid_num):
            valid_datas.append(normal_datas[i])
            valid_labels.append(0)
        elif i < total_len:
            test_datas.append(normal_datas[i])
            test_labels.append(0)
        else:
            break

    train_df['Sequence'] = train_datas
    train_df['label'] = train_labels
    valid_df['Sequence'] = valid_datas
    valid_df['label'] = valid_labels
    test_df['Sequence'] = test_datas
    test_df['label'] = test_labels
    # log_param('length of train_df',len(train_df))
    # log_param('length of valid_df',len(valid_df))
    # log_param('length of test_df',len(test_df))
    print(len(train_df))
    print(len(valid_df))
    print(len(test_df))
    train_df.to_csv(train_file,index=0)
    valid_df.to_csv(valid_file,index=0)
    test_df.to_csv(test_file,index=0)


def train_test_gen_4_smallTB(normal_in_file, abnormal_in_file, train_ratio, output_root_path):
    normal_datas = data_read(normal_in_file)
    abnormal_datas = data_read(abnormal_in_file)
    train_num = 9000
    valid_num = 10000
    train_file = output_root_path +'robust_log_train.csv'
    valid_file = output_root_path +'robust_log_valid.csv'
    test_file = output_root_path +'robust_log_test.csv'
    normal_datas = shuffle(normal_datas)
    abnormal_datas = shuffle(abnormal_datas)
    train_datas = []
    valid_datas = []
    test_datas = []
    train_labels = []
    valid_labels = []
    test_labels = []
    train_df = pd.DataFrame(columns=['Sequence','label'])
    valid_df = pd.DataFrame(columns=['Sequence','label'])
    test_df = pd.DataFrame(columns=['Sequence','label'])
    for i in range(len(abnormal_datas)):
        if i < train_num:
            train_datas.append(abnormal_datas[i])
            train_labels.append(1)
        elif (i >= train_num) & (i < valid_num):
            valid_datas.append(abnormal_datas[i])
            valid_labels.append(1)
        else:
            test_datas.append(abnormal_datas[i])
            test_labels.append(1)
    
    for i in range(len(normal_datas)):
        if i < train_num:
            train_datas.append(normal_datas[i])
            train_labels.append(0)
        elif (i >= train_num) & (i < valid_num):
            valid_datas.append(normal_datas[i])
            valid_labels.append(0)
        else:
            test_datas.append(normal_datas[i])
            test_labels.append(0)

    train_df['Sequence'] = train_datas
    train_df['label'] = train_labels
    valid_df['Sequence'] = valid_datas
    valid_df['label'] = valid_labels
    test_df['Sequence'] = test_datas
    test_df['label'] = test_labels
    # log_param('length of train_df',len(train_df))
    # log_param('length of valid_df',len(valid_df))
    # log_param('length of test_df',len(test_df))
    train_df.to_csv(train_file,index=0)
    valid_df.to_csv(valid_file,index=0)
    test_df.to_csv(test_file,index=0)

def train_test_gen_4_bgl(normal_in_file, abnormal_in_file, train_ratio, output_root_path):
    normal_datas = data_read(normal_in_file)
    abnormal_datas = data_read(abnormal_in_file)
    train_num = int(train_ratio*len(abnormal_datas))
    valid_num = train_num + int(0.05*len(abnormal_datas))
    train_file = output_root_path +'robust_log_train.csv'
    valid_file = output_root_path +'robust_log_valid.csv'
    test_file = output_root_path +'robust_log_test.csv'
    normal_datas = shuffle(normal_datas)
    abnormal_datas = shuffle(abnormal_datas)
    train_datas = []
    valid_datas = []
    test_datas = []
    train_labels = []
    valid_labels = []
    test_labels = []
    train_df = pd.DataFrame(columns=['Sequence','label'])
    valid_df = pd.DataFrame(columns=['Sequence','label'])
    test_df = pd.DataFrame(columns=['Sequence','label'])
    for i in range(len(abnormal_datas)):
        if i < train_num:
            train_datas.append(abnormal_datas[i])
            train_labels.append(1)
        elif (i >= train_num) & (i < valid_num):
            valid_datas.append(abnormal_datas[i])
            valid_labels.append(1)
        else:
            test_datas.append(abnormal_datas[i])
            test_labels.append(1)
    
    for i in range(len(normal_datas)):
        if i < train_num*7:
            train_datas.append(normal_datas[i])
            train_labels.append(0)
        elif (i >= train_num*7) & (i < valid_num*7):
            valid_datas.append(normal_datas[i])
            valid_labels.append(0)
        else:
            test_datas.append(normal_datas[i])
            test_labels.append(0)

    train_df['Sequence'] = train_datas
    train_df['label'] = train_labels
    valid_df['Sequence'] = valid_datas
    valid_df['label'] = valid_labels
    test_df['Sequence'] = test_datas
    test_df['label'] = test_labels
    # log_param('length of train_df',len(train_df))
    # log_param('length of valid_df',len(valid_df))
    # log_param('length of test_df',len(test_df))
    train_df.to_csv(train_file,index=0)
    valid_df.to_csv(valid_file,index=0)
    test_df.to_csv(test_file,index=0)


if __name__ == "__main__":
    parser = 'Lenma'
    dataset = 'HDFS'
    version = 'w18_s10_seven'
    # log_param('dataset',dataset)
    # log_param('parser',parser)
    # normal_in_file = '/nas/fuying/Data_for_time_test/'+dataset+'/'+parser+'/normal_abnormal_w18_s10/normal_temp.txt'
    # abnormal_in_file = '/nas/fuying/Data_for_time_test/'+dataset+'/'+parser+'/normal_abnormal_w18_s10/abnormal_temp.txt'
    normal_in_file = '/nas/fuying/Data_for_time_test/BGL/ground_truth/normal_abnormal/normal_temp.txt'
    abnormal_in_file = '/nas/fuying/Data_for_time_test/BGL/ground_truth/normal_abnormal/abnormal_temp.txt'
    train_ratio = 0.8
    # output_root_path = '/nas/fuying/Data_for_time_test/'+dataset+'/'+parser+'/train_test_data_4_robust/'+version+'/'
    output_root_path = '/nas/fuying/Data_for_time_test/BGL/ground_truth/train_test_data_4_robust/w18_s10_seven_v2'
    if not os.path.exists(output_root_path):
      os.makedirs(output_root_path)
    train_test_gen_4_hdfs(normal_in_file, abnormal_in_file, train_ratio, output_root_path)
    # train_test_gen(normal_in_file, abnormal_in_file, train_ratio, output_root_path)
    # train_test_gen_4_bgl(normal_in_file, abnormal_in_file, train_ratio, output_root_path)