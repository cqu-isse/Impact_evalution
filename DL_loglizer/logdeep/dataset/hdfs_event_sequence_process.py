"""
The interface for hdfs data preprocessing.
mainly for event sequence process.(for session)

Authors: FuYing
Date: 2020-09-16
"""

import pandas as pd
from random import sample
from tqdm import tqdm
import re
from collections import OrderedDict

class hdfs_DataTool():
    def __init__(self):
        pass

    def add_blockId_columns(self, logData):
        """
        Args:
        logData:从文件中读入的解析好的日志内容
        func:
        从Content列解析出BlockId，并添加列
        """
        blkId_list = []
        for content in logData['Content']:
            blkId = re.findall(r'(blk_-?\d+)', content)
            blkId_list.append(blkId[0])
        logData['blkId'] = blkId_list
        return logData

    def event_sequence_transform(self, data_dict, log_template_file):
        """
        Args:
        data_dict:以字典暂存的事件序列，序列为事件ID构成
        log_template_file:事件模版，用于将事件ID转为事件索引
        func:
        将事件序列中的事件ID转为事件索引
        """

        log_templates = pd.read_csv(log_template_file, usecols=['EventId'])
        log_template_dict = {}
        # #事件索引编号，从0开始
        # for index, event in enumerate(log_templates['EventId'].tolist()):
        #     log_template_dict[event] = index

        #事件索引编号，从1开始
        for index, event in enumerate(log_templates['EventId'].tolist()):
            log_template_dict[event] = index+1
        
        #将事件ID转为事件索引
        print('=======事件序列中，事件ID转为事件索引=======')
        for key in tqdm(data_dict.keys(), total=len(data_dict.keys())):
            for i in range(len(data_dict[key])):
                data_dict[key][i] = log_template_dict[data_dict[key][i]]
                    # data_dict[key][i] = 200
        return data_dict

    def event_by_block(self, input_path, log_template_file, event_output_path):
        """
        Args:
        input_path:解析好的hdfs数据的存储路径,文件格式要求为.csv文件
        log_template_file:日志模版文件
        event_output_path:存储按照BlockId分组并生成事件序列数据的存储路径
        func:
        将hdfs数据按照BlockId分组,产生事件序列和组件序列
        缺点:组件序列和事件序列同时生成，存在不灵活性。如果只想要重跑其中一个序列，函数中必须两个一起重跑。
        """
        logData = pd.read_csv(input_path, na_filter=False, header='infer')
        #解析出BlockId，并添加在最后一列
        logData = self.add_blockId_columns(logData)

        #按BlockId分组
        log_groups = logData.groupby('blkId')

        #以字典形式暂存产生的事件序列
        event_seq_dict = OrderedDict()
        print('=======事件序列生成=======')
        for blockid, group in tqdm(log_groups, total=logData['blkId'].drop_duplicates().shape[0]):
            if not blockid in event_seq_dict:
                event_seq_dict[blockid] = []
            event_seq_dict[blockid]=list(group['EventId'])

        
        #将产生的事件序列，通过事件模版，将事件ID转为以事件索引序号，方便后面建模
        event_seq_dict = self.event_sequence_transform(event_seq_dict, log_template_file)


        #写出事件序列
        event_data_df = pd.DataFrame(list(event_seq_dict.items()), columns=['BlockId', 'EventSequence'])
        event_data_df.to_csv(event_output_path, index=False)


    def data_to_line_by_events(self,label_input_path, event_sequence_path, output_file_folder):
        """
        Args:
        label_input_path:标签数据的存储路径,文件格式要求为.csv文件
        event_sequence_path:处理好的事件序列数据存储文件，文件格式要求为.csv文件
        output_path:存储按照BlockId添加完标签的数据
        func:
        将hdfs数据将日志模板编号按照BlockId逐行写入文本文件，将数据转化为Event序列
        """
        labelData = pd.read_csv(label_input_path, na_filter=False, header='infer', memory_map=True)
        event_sequence_data = pd.read_csv(event_sequence_path, header='infer')
        #使用BlockId作为index
        labelData = labelData.set_index('BlockId')
        label_dict = labelData['Label'].to_dict()

        #根据BlockId添加label
        event_sequence_data['Label'] = event_sequence_data['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)
        EventSequences = list(event_sequence_data['EventSequence'])
        labels = list(event_sequence_data['Label'])
        for i in range(len(EventSequences)):
            #取出窗口中的事件list,并将事件list转为事件序列字符串
            event_seq = str(EventSequences[i]).replace('[','').replace(']','').replace(',','').replace('\'','')+'\n'
            if labels[i] == 1:
                with open(output_file_folder+'abnormal_temp.txt','a+') as f:
                    f.writelines(event_seq)
            else:
                with open(output_file_folder+'normal_temp.txt','a+') as f:
                    f.writelines(event_seq)


        # event_sequence_data.to_csv(output_path, index=False)


def test():
    datatool = hdfs_DataTool()

    dataset = 'HDFS'
    parser = 'LFA'

    # input_file = '/home/fuying/logparser/demo/'+parser+'_result/'+dataset+'.log_structured_process.csv'
    # log_template_file = '/home/fuying/logparser/demo/'+parser+'_result/'+dataset+'.log_templates.csv'
    # output_file= '/home/fuying/data_hub/for_deeplog/'+dataset+'_log/parsing_by_'+parser+'/'+dataset+'_all_data_labeled.csv'
    # datatool.event_by_block(input_file,log_template_file, output_file)

    input_file_labeled = '/home/fuying/data_hub/HDFS_log/anomaly_label.csv'
    event_sequence_path = '/home/fuying/data_hub/for_deeplog/'+dataset+'_log/parsing_by_'+parser+'/'+dataset+'_all_data_labeled.csv'
    output_file_folder = '/home/fuying/data_hub/for_deeplog/'+dataset+'_log/parsing_by_'+parser+'/normal_abnormal/'
    datatool.data_to_line_by_events(input_file_labeled,event_sequence_path, output_file_folder)


if __name__ == "__main__":
    test()

