#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
modified by: FuYing
date:2020.11.10
function: 添加固定窗口函数
'''

import json
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import mlflow
from mlflow import log_param, log_metrics, log_artifacts

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("rubustlog_experiment")

def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


def trp(l, n):
    """ Truncate or pad a list """
    r = l[:n]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r


def down_sample(logs, labels, sample_ratio):
    print('sampling...')
    total_num = len(labels)
    all_index = list(range(total_num))
    sample_logs = {}
    for key in logs.keys():
        sample_logs[key] = []
    sample_labels = []
    sample_num = int(total_num * sample_ratio)

    for i in tqdm(range(sample_num)):
        random_index = int(np.random.uniform(0, len(all_index)))
        for key in logs.keys():
            sample_logs[key].append(logs[key][random_index])
        sample_labels.append(labels[random_index])
        del all_index[random_index]
    return sample_logs, sample_labels

def sliding_window(data_dir, datatype, window_size, template_num, sample_ratio=1):
    '''
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    '''
    event2semantic_vec = read_json(data_dir + '/event2semantic_vec.json')
    num_sessions = 0
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    labels = []

    if datatype == 'train':
        data_dir += '/robust_log_train.csv'
    elif datatype == 'val':
        data_dir += '/robust_log_valid.csv'
    elif datatype == 'test':
        data_dir += '/robust_log_test.csv'

    with open(data_dir, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))

            for i in range(len(line) - window_size):
                Sequential_pattern = list(line[i:i + window_size])
                Quantitative_pattern = [0] * template_num
                log_counter = Counter(Sequential_pattern)

                for key in log_counter:
                    Quantitative_pattern[key] = log_counter[key]

                Semantic_pattern = []
                for event in Sequential_pattern:
                    if event == 0:
                        Semantic_pattern.append([-1] * 300)
                    else:
                        Semantic_pattern.append(event2semantic_vec[str(event -
                                                                       1)])
                Sequential_pattern = np.array(Sequential_pattern)[:,
                                                                  np.newaxis]
                Quantitative_pattern = np.array(
                    Quantitative_pattern)[:, np.newaxis]

                result_logs['Sequentials'].append(Sequential_pattern)
                result_logs['Quantitatives'].append(Quantitative_pattern)
                result_logs['Semantics'].append(Semantic_pattern)
                labels.append(line[i + window_size])

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    print('File {}, number of sessions {}'.format(data_dir, num_sessions))
    print('File {}, number of seqs {}'.format(data_dir,
                                              len(result_logs['Sequentials'])))

    return result_logs, labels

def sliding_window_v2(data_dir, datatype, window_size, sample_ratio=1):
    '''
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    '''
    event2semantic_vec = read_json(data_dir + '/event2semantic_vec.json')
    num_sessions = 0
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    labels = []
    if datatype == 'train':
        data_dir += '/robust_log_train.csv'
    elif datatype == 'val':
        data_dir += '/robust_log_valid.csv'
    elif datatype == 'test':
        data_dir += '/robust_log_test.csv'


    with open(data_dir, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))

            for i in range(len(line) - window_size):
                Sequential_pattern = list(line[i:i + window_size])
                Quantitative_pattern = [0] * 28
                log_counter = Counter(Sequential_pattern)

                for key in log_counter:
                    Quantitative_pattern[key] = log_counter[key]
                Semantic_pattern = []
                for event in Sequential_pattern:
                    if event == 0:
                        Semantic_pattern.append([-1] * 300)
                    else:
                        Semantic_pattern.append(event2semantic_vec[str(event -
                                                                       1)])
                Sequential_pattern = np.array(Sequential_pattern)[:,
                                                                  np.newaxis]
                Quantitative_pattern = np.array(
                    Quantitative_pattern)[:, np.newaxis]
                result_logs['Sequentials'].append(Sequential_pattern)
                result_logs['Quantitatives'].append(Quantitative_pattern)
                result_logs['Semantics'].append(Semantic_pattern)
                labels.append(line[i + window_size])

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    print('File {}, number of sessions {}'.format(data_dir, num_sessions))
    print('File {}, number of seqs {}'.format(data_dir,
                                              len(result_logs['Sequentials'])))

    return result_logs, labels


def fixed_window(data_dir, datatype, window_size, template_num, sample_ratio=1):
    '''
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    '''
    event2semantic_vec = read_json(data_dir + '/event2semantic_vec.json')
    num_sessions = 0
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    labels = []
    if datatype == 'train':
        data_dir += '/robust_log_train.csv'
    elif datatype == 'val':
        data_dir += '/robust_log_valid.csv'
    elif datatype == 'test':
        data_dir += '/robust_log_test.csv'
    train_df = pd.read_csv(data_dir)
    for j in tqdm(range(len(train_df))):
        num_sessions += 1
        line = [int(eventid) for eventid in train_df["Sequence"][j].split(' ')]
        Sequential_pattern = trp(line, window_size)
        Semantic_pattern = []
        for event in Sequential_pattern:
            if event == 0:
                Semantic_pattern.append([-1] * 300)
            else:
                Semantic_pattern.append(event2semantic_vec[str(event - 1)])
        Quantitative_pattern = [0] * template_num
        log_counter = Counter(Sequential_pattern)

        for key in log_counter:
            Quantitative_pattern[key] = log_counter[key]

        Sequential_pattern = np.array(Sequential_pattern)[:,np.newaxis]
        Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
        result_logs['Sequentials'].append(Sequential_pattern)
        result_logs['Quantitatives'].append(Quantitative_pattern)
        result_logs['Semantics'].append(Semantic_pattern)
        labels.append(int(train_df["label"][j]))

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    # log_param('input file dir',data_dir)
    #number of sessions要与number of seqs相同
    # log_param('number of sessions',num_sessions)
    # log_param('number of seqs',len(result_logs['Semantics']))
    print('File {}, number of sessions {}'.format(data_dir, num_sessions))
    print('File {}, number of seqs {}'.format(data_dir,
                                              len(result_logs['Semantics'])))

    return result_logs, labels



def session_window(data_dir, datatype, template_num, sample_ratio=1):
    event2semantic_vec = read_json(data_dir + '/event2semantic_vec.json')
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    labels = []

    if datatype == 'train':
        data_dir += '/robust_log_train.csv'
    elif datatype == 'val':
        data_dir += '/robust_log_valid.csv'
    elif datatype == 'test':
        data_dir += '/robust_log_test.csv'

    train_df = pd.read_csv(data_dir)
    for i in tqdm(range(len(train_df))):
        ori_seq = [
            int(eventid) for eventid in train_df["Sequence"][i].split(' ')
        ]
        Sequential_pattern = trp(ori_seq, 50)
        Semantic_pattern = []
        for event in Sequential_pattern:
            if event == 0:
                Semantic_pattern.append([-1] * 300)
            else:
                Semantic_pattern.append(event2semantic_vec[str(event - 1)])
        Quantitative_pattern = [0] * template_num
        log_counter = Counter(Sequential_pattern)
        for key in log_counter:
            if key in range(len(Quantitative_pattern)):
                Quantitative_pattern[key] = log_counter[key]
            else:
                print(Quantitative_pattern)
                print(log_counter)

        Sequential_pattern = np.array(Sequential_pattern)[:, np.newaxis]
        Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
        result_logs['Sequentials'].append(Sequential_pattern)
        result_logs['Quantitatives'].append(Quantitative_pattern)
        result_logs['Semantics'].append(Semantic_pattern)
        labels.append(int(train_df["label"][i]))

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)
    print('Number of sessions({}): {}'.format(data_dir,
                                              len(result_logs['Semantics'])))
    return result_logs, labels

