#!/usr/bin/env python
import sys
sys.path.append('../')
# sys.path.append('/home/fuying/logparser/logparser/')
from logparser.LenMa import LenMa
# from logparser import LenMa
import os
Project_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

input_dir = os.path.join(Project_path, "logparser/log_data/HDFS") # The input directory of log file
output_dir = os.path.join(Project_path, "logparser/parsing_result/HDFS/Lenma_result/")  # The output directory of parsing results
log_file   = 'HDFS.log' # The input log file name
log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>' # HDFS log format
threshold  = 0.9 # TODO description (default: 0.9)
regex      = [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']  # Regular expression list for optional preprocessing (default: [])

parser = LenMa.LogParser(input_dir, output_dir, log_format, threshold=threshold, rex=regex)
parser.parse(log_file)
