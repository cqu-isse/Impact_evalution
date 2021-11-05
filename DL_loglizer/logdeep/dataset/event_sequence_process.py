"""
The interface for log data preprocessing.
mainly for event sequence process(for windows not session).

Authors: FuYing
Date: 2020-10-26
"""


import pandas as pd
from random import sample
from tqdm import tqdm
import re
from collections import OrderedDict
import os

class DataTool():
    def __init__(self):
        pass

    def label_template(self, input_file, templates_file, output_file):
        """
        Args:
        input_file:解析好的数据文件
        template_file:解析得到的模版文件
        output_file:处理结果保存文件
        func:
        对事件模版编号，并标记模版标签。将中间处理结果写出，节省处理成其他格式输入文件的时间。
        """
        aim_columns = ['Label', 'Timestamp', 'EventId', 'Component']
        input_data = pd.read_csv(input_file, chunksize=10000, usecols=aim_columns)
        logtemp_data = pd.read_csv(templates_file, usecols=['EventId'])
        logtemp_dict = {}

        for index, event in enumerate(logtemp_data['EventId']):
            logtemp_dict[event] = index + 1

        if os.path.exists(output_file):
            os.remove(output_file)
        rr = []
        for index, input_data in enumerate(tqdm(input_data, total=480)):
            input_data['EventIdNum'] = None
            #将解析出的事件号，映射为模版事件编号
            for i in input_data.index:
                event_id_by_parsing = str(input_data.loc[i,'EventId'])
                if event_id_by_parsing in logtemp_dict.keys():
                    input_data.loc[i, 'EventIdNum'] = logtemp_dict[event_id_by_parsing]
                elif event_id_by_parsing == '4319570':
                    input_data.loc[i, 'EventIdNum'] = logtemp_dict['04319570']

            #写出结果
            if index != 0:
                input_data.to_csv(output_file, mode = 'a+', index = False, header = False)
            else:
                input_data.to_csv(output_file, mode = 'a+', index = False)

    def data_to_line_by_window(self, input_file_labeled, window_size, step, output_file_folder):
        """
        Args:
        input_file_labeled:对事件模版编号，并标记好模版标签的文件
        window_size:窗口大小
        step:步长
        output_file:处理结果保存文件（.txt）
        func:
        将数据按照切分规则（划动窗口）逐行写入txt文件，一个事件序列的标签由内部事件的标签决定。
        """
        labelData = pd.read_csv(input_file_labeled, header='infer')
        print('=========================')
        print(labelData.shape[0])
        if os.path.exists(output_file_folder+'normal_temp.txt'):
            os.remove(output_file_folder+'normal_temp.txt')
        if os.path.exists(output_file_folder+'abnormal_temp.txt'):
            os.remove(output_file_folder+'abnormal_temp.txt')
        count = 0  
        for x in tqdm(range(int(labelData.shape[0]/window_size))):
            #取窗口中的子序列数据
            sub_seq_data = labelData.loc[x*window_size:(x+1)*window_size-1]
            #取出窗口中的事件list,并将事件list转为事件序列字符串
            event_seq = str(sub_seq_data['EventIdNum'].tolist()).replace('[','').replace(']','').replace(',','').replace('\'','')+'\n'
            if len(set(sub_seq_data['Label'].tolist())) == 1:
                with open(output_file_folder+'normal_temp.txt','a+') as f:
                    f.writelines(event_seq)
            else:
                count +=1
                with open(output_file_folder+'abnormal_temp.txt','a+') as f:
                    f.writelines(event_seq)
        print('abnormal count ======>', count)

def data_eval(dataset,parser):
    in_file = '/home/fuying/data_hub/for_deeplog/'+dataset+'_log/parsing_by_'+parser+'/'+dataset+'_all_data_labeled.csv'
    labelData = pd.read_csv(in_file, header='infer')
    for event in list(labelData['EventIdNum']):
        if event == 0:
            print(event)

def test():
    datatool = DataTool()
    dataset = 'BGL'
    parser = 'Drain'
    # data_eval(dataset,parser)

    # input_file = '/home/fuying/logparser/demo/'+parser+'_result/'+dataset+'.log_structured_process.csv'
    # log_template_file = '/home/fuying/logparser/demo/'+parser+'_result/'+dataset+'.log_templates.csv'
    # output_file= '/home/fuying/data_hub/for_deeplog/'+dataset+'_log/parsing_by_'+parser+'/'+dataset+'_all_data_labeled.csv'
    # datatool.label_template(input_file,log_template_file, output_file)

    input_file_labeled = '/home/fuying/data_hub/'+dataset+'_log/parsing_by_'+parser+'/'+dataset+'_all_data_labeled.csv'
    window_size = 50
    step = 20
    output_file_folder = '/home/fuying/data_hub/'+dataset+'_log/parsing_by_'+parser+'/normal_abnormal_v6/'
    datatool.data_to_line_by_window(input_file_labeled, window_size, step, output_file_folder)

def normal_abnormal_split():
    datatool = DataTool()
    dataset = 'BGL'
    window_size = 18
    step = 5
    # parsers = ['Drain','IPLoM','Logram','LFA','Lenma','Spell']
    parsers = ['Drain']
    for parser in parsers:
        input_file_labeled = '/nas/fuying/Data_for_time_test/'+dataset+'/'+parser+'/'+dataset+'_all_data_labeled.csv'
        output_file_folder = '/nas/fuying/Data_for_time_test/'+dataset+'/'+parser+'/normal_abnormal_w18_s5/'
        if not os.path.exists(output_file_folder):
            os.makedirs(output_file_folder)
        datatool.data_to_line_by_window(input_file_labeled, window_size, step, output_file_folder)


if __name__ == "__main__":
    normal_abnormal_split()

 