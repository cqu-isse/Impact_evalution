#!/usr/bin/env python

import sys
sys.path.append('../')
import evaluator
import os
import pandas as pd


def test():
    parser = 'LFA'
    parsing_file = '/home/fuying/logparser/demo/'+parser+'_result/BGL.log_structured_process.csv'
    groundtruth_file = '/nas/fuying/Data_for_time_test/BGL/ground_truth/BGL_all_data_labeled.csv'
    F1_measure, accuracy = evaluator.evaluate(groundtruth=groundtruth_file,parsedresult=parsing_file)
    print(parser)
    print(accuracy)
    print(F1_measure)

def tt():
    in_file1 = '/nas/fuying/Source_code/logparser-master/demo/logmatch_result/BGL.log_templates.csv'
    in_file2 = '/nas/fuying/Data_for_time_test/BGL/ground_truth/BGL.log_templates.csv'
    contents2 = pd.read_csv(in_file2)
    contents1 = pd.read_csv(in_file1)
    contents1_event = list(contents1['EventId'])
    contents2_event = list(contents2['EventId'])
    for event in contents2_event:
        if event not in contents1_event:
            print(event)


def event_search(groundtruth, parsedresult):
    """ Evaluation function to benchmark log parsing accuracy
    
    Arguments
    ---------
        groundtruth : str
            file path of groundtruth structured csv file 
        parsedresult : str
            file path of parsed structured csv file
    """ 
    df_groundtruth = pd.read_csv(groundtruth)
    df_parsedlog = pd.read_csv(parsedresult)
    # Remove invalid groundtruth event Ids
    null_logids = df_groundtruth[~df_groundtruth['EventId'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedlog = df_parsedlog.loc[null_logids]

    series_groundtruth = df_groundtruth['EventId']
    series_parsedlog = df_parsedlog['EventId']

    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    
    for groundtruth_eventId in series_groundtruth_valuecounts.index:
        groundtruth_logIds = series_groundtruth[series_groundtruth == groundtruth_eventId].index
        parsed_event = series_parsedlog[groundtruth_logIds]
        if len(set(parsed_event))>200:
            print(groundtruth_eventId)
            print(len(set(parsed_event)))
            # print(set(parsed_event))
            print('====================================')

groundtruth = '/nas/fuying/Data_for_time_test/BGL/ground_truth/BGL_all_data_labeled.csv'
parsedresult = '/home/fuying/logparser/demo/Spell_result/BGL.log_structured.csv'
event_search(groundtruth, parsedresult)

# test()