"""
Description : This file implements the Logram algorithm for log parsing
Author      : FuYing
"""

import re
import os
import numpy as np
import pandas as pd
import hashlib
from datetime import datetime
# from MatchToken import *
# from log_to_df import *
# from DictionarySetUp import *


from .DictionarySetUp import dictionaryBuilder_fy
from .MatchToken import tokenMatch_fy

class LogParser:
    def __init__(self,log_format, indir='./', outdir='./result/', doubleThreshold=10, triThreshold=4,rex=[], keep_para=True):
        """
        Attributes
        ----------
            rex : regular expressions used in preprocessing (step1)
            path : the input path stores the input log file name
            logName : the name of the input file containing raw log messages
            savePath : the output path stores the file containing structured logs
        """
        self.path=indir
        self.doubleThreshold = doubleThreshold
        self.triThreshold = triThreshold
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex
        self.keep_para = keep_para

    def parse(self, logName, fold_num):
        log_file_path = os.path.join(self.path, logName)
        print('Parsing file: ' + log_file_path)
        start_time = datetime.now()
        self.logName = logName

        if not os.path.exists(self.savePath):  #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(self.savePath)
        out_path = os.path.join(self.savePath, logName)
        # print('out_path: ' + out_path)
        
        doubleDictionaryList, triDictionaryList, allTokenList, log_contents = dictionaryBuilder_fy(self.log_format, log_file_path, self.rex)
        tokenMatch_fy(log_contents, allTokenList, doubleDictionaryList, triDictionaryList,self.doubleThreshold, self.triThreshold, out_path)
        log_parser = 'Logram'
