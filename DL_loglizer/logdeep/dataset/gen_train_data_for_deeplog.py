import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def train_test_split_for_deeplog(input_file, train_ratio, output_root_path):
    input_data_num = len(np.loadtxt(input_file))
    # input_data_num = 558223
    print('==============input_data_num=============')
    print(input_data_num)
    train_num = int(train_ratio*input_data_num)

    train_file = output_root_path +'normal_train.txt'
    test_file = output_root_path +'normal_test.txt'

    n = 0
    with open(input_file,"r", encoding='UTF-8') as f:
        with open(train_file, "w", encoding='UTF-8') as f1:
            with open(test_file, "w", encoding='UTF-8') as f2:
                for line in tqdm(f.readlines(),total = input_data_num):
                    n = n + 1
                    if n <= train_num:
                        f1.writelines(line)
                    else:
                        f2.writelines(line)

def data_split():
    dataset = 'smallTB'
    parser = 'Lenma'
    input_file = '/home/fuying/data_hub/for_deeplog/'+dataset+'_log/parsing_by_'+parser+'/normal_abnormal/normal_temp.txt'
    train_ratio = 0.8
    output_root_path = '/home/fuying/data_hub/for_deeplog/'+dataset+'_log/parsing_by_'+parser+'/normal_abnormal/'
    train_test_split_for_deeplog(input_file, train_ratio, output_root_path)

if __name__ == "__main__":
    data_split()