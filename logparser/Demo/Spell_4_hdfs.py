#!/usr/bin/env python
import sys
sys.path.append('../')
from logparser import Spell

input_dir  = '/home/fuying/data_hub/HDFS_log/'  # The input directory of log file
output_dir = 'Spell_result/'  # The output directory of parsing results
log_file   = 'HDFS.log'  # The input log file name
log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
tau        = 0.7  # Message type threshold (default: 0.5)
regex      = [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']  # Regular expression list for optional preprocessing (default: [])

parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
parser.parse(log_file)
