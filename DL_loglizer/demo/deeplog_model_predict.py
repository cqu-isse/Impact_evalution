"""
for predict

Authors: FuYing
Date: 2020-09-21
"""
import sys
sys.path.append('../')
import torch
import torch.nn as nn

import argparse
from tqdm import tqdm
import os
from logdeep.models.logC_model import *

from datetime import datetime

from logdeep.tools.utils import load_words, write_out


# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Hyperparameters
input_size = 1
num_class_dic = {'Drain':48,'IPLoM':45,'Spell':37, 'Logram':97, 'LFA':47, 'Lenma':45, 'GroundTruth':29}
log_parser = 'Logram'
dataset = 'HDFS'
num_classes = num_class_dic[log_parser]
window_size = 10
model_dir = '/home/fuying/data_hub/model_hub/deeplog_'+log_parser+'_4_'+ dataset+'_v6/Log_Adam_batch_size=256_epoch=300'

def generate(name):
    contents =[]
    with open('/home/fuying/data_hub/'+ dataset + '_log/parsing_by_'+log_parser+'/normal_abnormal_v6/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(int, ln.strip().split()))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            contents.append(tuple(ln))
    print('Number of sessions({}): {}'.format(name, len(contents)))
    return contents

def model_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates

    model = logKey_model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    print('model_path: {}'.format(model_dir))
    test_normal_loader = generate('normal_test.txt')
    test_abnormal_loader = generate('abnormal_temp.txt')
    TP = 0
    FP = 0

    FP_datas = []
    FP_datas_path = '/nas/fuying/Data_for_time_test/HDFS/'+log_parser+'/FP_datas_v6.txt'
    # Test the model
    normal_results = []
    with torch.no_grad():
        for line in tqdm(test_normal_loader, desc="normal_test"):
            normal_flag = 0
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                out_label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    FP += 1
                    line = list(line)
                    line.append(i)
                    ln = list(map(str, line))
                    FP_datas.append(' '.join(ln)+' '+ str(out_label))
                    normal_flag = 1
                    break

            if normal_flag == 0:
                line = list(line)
                ln = list(map(str, line))
                normal_results.append(' '.join(ln)+',Y')
            else:
                line = list(line)
                ln = list(map(str, line))
                normal_results.append(' '.join(ln)+',N')

    print('==================')
    print(len(FP_datas))
    print(FP)
    print('==================')
    
    TN = len(test_normal_loader) - FP

    TP_datas = []
    TP_datas_path = '/nas/fuying/Data_for_time_test/HDFS/'+log_parser+'/TP_datas_v6.txt'
    TP_count = -1
    TP_count_list = []
    
    abnormal_results = []
    with torch.no_grad():
        for line in tqdm(test_abnormal_loader, desc="abnormal_temp"):
            abnormal_flag = 0
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    TP += 1
                    line = list(line)
                    line.append(i)
                    ln = list(map(str, line))
                    TP_datas.append(' '.join(ln))
                    TP_count_list.append(TP_count)
                    abnormal_flag = 1
                    break

            if abnormal_flag == 0:
                line = list(line)
                ln = list(map(str, line))
                abnormal_results.append(' '.join(ln)+',N')
            else:
                line = list(line)
                ln = list(map(str, line))
                abnormal_results.append(' '.join(ln)+',Y')

    print('==================')
    print(len(TP_datas))
    FN = len(test_abnormal_loader) - TP
    print(TP)
    print('==================')
    print("FP = {},FN = {},TP = {}, TN = {}".format(FP, FN, TP, TN))

    # Compute precision, recall and F1-measure
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(P, R, F1))
    log_metrics({'FP':FP,'FN':FN,'TP':TP,'TN':TN,'P':P,'R':R,'F1':F1})
    print('Finished Predicting')
    write_out(FP_datas,'\n',FP_datas_path)
    write_out(TP_datas,'\n',TP_datas_path)

    normal_results_outfile = '/nas/fuying/Data_for_time_test/HDFS/'+log_parser+'/Diff/v6_normal_results.csv'
    abnormal_results_outfile = '/nas/fuying/Data_for_time_test/HDFS/'+log_parser+'/Diff/v6_abnormal_results.csv'
    write_out(normal_results,'\n',normal_results_outfile)
    write_out(abnormal_results,'\n',abnormal_results_outfile)


if __name__ == "__main__":
    model_predict()
