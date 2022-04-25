1. logparser is used for log parsing
2. ML_loglizer: machine learning based anomaly detection
3. DL_loglizer: deep learning based anomaly detection

# Setup
Python ==3.6  

numpy ==1.19.5

sklearn == 0.24.2

torth == 1.7.1

scipy == 1.5.2
#### Dataset
We use the dataset of Logpai (https://github.com/logpai/loghub)

# Start
1. Dataset
Download the log data and put the log data in the corresponding folder under log_data

2. Parsing log data 
For example: 
python Drain_4_hdfs.py

Note: The data parsing by LFA needs to be processed by LFA_data_process.py

3. Train and evaluate your anomal detection model
For example:
python DecisionTree_4_HDFS.py