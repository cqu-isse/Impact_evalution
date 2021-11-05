from .Common import regexGenerator
from .Common import tokenSpliter
from .log_to_df import contents_to_df

def dictionaryBuilder(log_format, logFile, rex):
    doubleDictionaryList = {'dictionary^DHT': -1};
    triDictionaryList = {'dictionary^DHT^triple': -1};
    allTokenList = []

    regex = regexGenerator(log_format)

    for line in open(logFile, 'r', encoding="utf-8", errors='ignore'):
        #print(line)
        tokens = tokenSpliter(line, regex, rex)
        if(tokens == None):
            pass;
        else:
            allTokenList.append(tokens)
            for index in range(len(tokens)):
                if index >= len(tokens) - 2:
                    break;
                tripleTmp = tokens[index] + '^' + tokens[index + 1] + '^' + tokens[index + 2];
                if tripleTmp in triDictionaryList:
                    triDictionaryList[tripleTmp] = triDictionaryList[tripleTmp] + 1;
                else:
                    triDictionaryList[tripleTmp] = 1;
            for index in range(len(tokens)):
                if index == len(tokens)-1:
                    break;
                doubleTmp = tokens[index] + '^' + tokens[index+1];
                if doubleTmp in doubleDictionaryList:
                    doubleDictionaryList[doubleTmp] = doubleDictionaryList[doubleTmp] + 1;
                else:
                    doubleDictionaryList[doubleTmp] = 1;
    return doubleDictionaryList, triDictionaryList, allTokenList



def dictionaryBuilder_fy(log_format, logFile, rex):
    doubleDictionaryList = {'dictionary^DHT': -1};
    triDictionaryList = {'dictionary^DHT^triple': -1};
    allTokenList = []

    regex = regexGenerator(log_format)
    log_contents = []

    for line in open(logFile, 'r', encoding="utf-8", errors='ignore'):
        #print(line)
        tokens = tokenSpliter(line, regex, rex)
        if(tokens == None):
            pass;
        else:
            allTokenList.append(tokens)
            log_contents.append(line.strip())
            for index in range(len(tokens)):
                if index >= len(tokens) - 2:
                    break;
                tripleTmp = tokens[index] + '^' + tokens[index + 1] + '^' + tokens[index + 2];
                if tripleTmp in triDictionaryList:
                    triDictionaryList[tripleTmp] = triDictionaryList[tripleTmp] + 1;
                else:
                    triDictionaryList[tripleTmp] = 1;
            for index in range(len(tokens)):
                if index == len(tokens)-1:
                    break;
                doubleTmp = tokens[index] + '^' + tokens[index+1];
                if doubleTmp in doubleDictionaryList:
                    doubleDictionaryList[doubleTmp] = doubleDictionaryList[doubleTmp] + 1;
                else:
                    doubleDictionaryList[doubleTmp] = 1;

    log_df = contents_to_df(log_contents, log_format, rex)
    return doubleDictionaryList, triDictionaryList, allTokenList, log_df