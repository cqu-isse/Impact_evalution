"""
The interface for model build.

Authors: FuYing
Date: 2020-09-16
"""
import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
from logdeep.models.logC_model import *
from datetime import datetime

Project_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Hyperparameters
window_size = 10
input_size = 1
hidden_size = 64
num_layers = 2
num_epochs = 300
batch_size = 256
log_parser = 'Logram'
dataset = 'HDFS'
num_class_dic = {'Drain':48,'IPLoM':45,'Spell':37, 'Logram':97, 'LFA':47, 'Lenma':45, 'GroundTruth':29}
num_classes = num_class_dic[log_parser]


model_dir = os.path.join(Project_path, 'DL_loglizer/model_hub/'+dataset+'/'+log_parser+'/')
log = 'Log_Adam_batch_size={}_epoch={}'.format(str(batch_size), str(num_epochs))

def generate_for_log_Key(name):
    num_sessions = 0
    inputs = []
    outputs = []
    dataset = 'HDFS'
    log_parser = 'Logram'
    file_root_path = os.path.join(Project_path, 'DL_loglizer/log_data/'+dataset+'/'+log_parser+'/normal_abnormal/')
    with open(file_root_path + name, 'r') as f:
        for line in f.readlines():
            line = tuple(map(int, line.strip().split()))
            if len(line) > window_size:
                num_sessions +=1
                for i in range(len(line) - window_size):
                    inputs.append(line[i:i + window_size])
                    outputs.append(line[i + window_size])
    print('Number of sessions({}): {}'.format(name, num_sessions))
    print('Number of seqs({}): {}'.format(name, len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset


def logKey_model_train():
    #网络参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size

    model = logKey_model(input_size, hidden_size, num_layers, num_classes).to(device)

    seq_dataset = generate_for_log_Key('normal_train.txt')
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    total_step = len(dataloader)
    starttime = datetime.now()
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
            # Forward pass
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            output = model(seq)
            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / total_step))
    torch.save(model.state_dict(), model_dir + log)
    # log_param('model training time', (datetime.now() - starttime))
    print('Finished Training. [Time taken: {!s}]'.format(datetime.now() - starttime))

if __name__ == "__main__":
    logKey_model_train()