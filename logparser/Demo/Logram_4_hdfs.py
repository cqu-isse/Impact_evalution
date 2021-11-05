#!/usr/bin/env python
import sys
sys.path.append('../')
sys.path.append('/home/fuying/logparser/logparser/')
from logparser.Logram import Logram

input_dir  = '/home/fuying/data_hub/HDFS_log/'# The input directory of log file
output_dir = 'Logram_result/' # The output directory of parsing results
log_file   = 'HDFS.log'  # The input log file name
log_format   = '<Date> <Time> <Pid> <Level> <Component>: <Content>' # hdfs log format
# Regular expression list for optional preprocessing (default: [])
regex      = [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']

doubleThreshold= 1840
triThreshold= 80
parser = Logram.LogParser(log_format,input_dir,output_dir,doubleThreshold,triThreshold,regex)
parser.parse(log_file)