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

Project_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))


# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Hyperparameters
input_size = 1
num_class_dic = {'Drain':48,'IPLoM':45,'Spell':37, 'Logram':97, 'LFA':47, 'Lenma':45, 'GroundTruth':29}
log_parser = 'Logram'
dataset = 'HDFS'
num_classes = num_class_dic[log_parser]
window_size = 10
model_dir = os.path.join(Project_path, 'DL_loglizer/model_hub/'+dataset+'/'+log_parser+'/Log_Adam_batch_size=256_epoch=300')
def generate(name):
    contents =[]
    dataset = 'HDFS'
    log_parser = 'Logram'
    file_root_path = os.path.join(Project_path, 'DL_loglizer/log_data/'+dataset+'/'+log_parser+'/normal_abnormal/')
    with open(file_root_path + name, 'r') as f:
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

    # Test the model
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
                    break

    
    TN = len(test_normal_loader) - FP

    
    with torch.no_grad():
        for line in tqdm(test_abnormal_loader, desc="abnormal_temp"):
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    TP += 1
                    break

    FN = len(test_abnormal_loader) - TP
    print("FP = {},FN = {},TP = {}, TN = {}".format(FP, FN, TP, TN))

    # Compute precision, recall and F1-measure
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(P, R, F1))
    # log_metrics({'FP':FP,'FN':FN,'TP':TP,'TN':TN,'P':P,'R':R,'F1':F1})
    print('Finished Predicting')


if __name__ == "__main__":
    model_predict()
