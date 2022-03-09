#!/usr/bin/env python
import sys
sys.path.append('../')
from logparser import Spell

import os
Project_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

input_dir = os.path.join(Project_path, "logparser/log_data/HDFS") # The input directory of log file
output_dir = os.path.join(Project_path, "logparser/parsing_result/HDFS/Spell_result/")  # The output directory of parsing results
log_file   = 'HDFS.log'  # The input log file name
log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
tau        = 0.7  # Message type threshold (default: 0.5)
regex      = [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']  # Regular expression list for optional preprocessing (default: [])

parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
parser.parse(log_file)
