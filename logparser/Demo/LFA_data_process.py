"""
The interface for log data preprocessing.
mainly for event sequence process(for windows not session).

Authors: FuYing
Date: 2021-04-23
"""


import pandas as pd
from random import sample
from tqdm import tqdm
import re
import os
Project_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

def event_mapping(input_file,log_template_file,output_file):
    input_data = pd.read_csv(input_file, chunksize=10000)
    logtemp_data = pd.read_csv(log_template_file, usecols=['EventId','EventTemplate'])
    logtemp_dict = {}

    for index, event in enumerate(logtemp_data['EventTemplate']):
        logtemp_dict[event] = logtemp_data.loc[index,'EventId']

    count = 0
    for index, input_data in enumerate(tqdm(input_data, total=480)):
        #将解析出的事件号，映射为模版事件编号
        for i in input_data.index:
            count += 1
            event_by_parsing = input_data.loc[i,'EventTemplate']
            # print(event_by_parsing)
            # print(logtemp_dict[event_by_parsing])
            input_data.loc[i, 'EventId'] = logtemp_dict[event_by_parsing]
            # print(input_data.loc[i, 'EventId'] )
            
        #写出结果
        if index != 0:
            input_data.to_csv(output_file, mode = 'a+', index = False, header = False)
        else:
            input_data.to_csv(output_file, mode = 'a+', index = False)
    print(count)



if __name__ == "__main__":
    dataset = 'HDFS_2k.log'
    parser = 'LFA'
    input_file = os.path.join(Project_path, 'logparser/parsing_result/'+dataset+'/'+parser+'_result/'+dataset+'_structured.csv') 
    log_template_file = os.path.join(Project_path, 'logparser/parsing_result/'+dataset+'/'+parser+'_result/'+dataset+'_templates.csv') 
    output_file = os.path.join(Project_path, 'logparser/parsing_result/'+dataset+'/'+parser+'_result/'+dataset+'_structured_process.csv')
    event_mapping(input_file,log_template_file,output_file)