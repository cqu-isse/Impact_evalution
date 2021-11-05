import re
import pandas as pd

def generate_logformat_regex(logformat):
    """ 
    Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex

def log_to_dataframe(log_contents, regex, headers, logformat):
    """ 
    Function to transform log file to dataframe 
    """
    log_messages = []
    linecount = 0
    for line in log_contents:
        try:
            match = regex.search(line.strip())
            message = [match.group(header) for header in headers]
            log_messages.append(message)
            linecount += 1
        except Exception as e:
            pass
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(linecount)]
    return logdf

def contents_to_df(log_contents, log_format, log_Regex):
    headers, regex = generate_logformat_regex(log_format)
    df_log = log_to_dataframe(log_contents, regex, headers, log_format)
    return df_log