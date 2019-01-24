### NER-BERT
***
#### 1. BiLSTMcrf 
&ensp;&ensp;基于Bi-LSTM + crf的中文命名实体识别
* 验证集上的结果：f1 = 0.835


#### 2. BERT
&ensp;&ensp;基于bert预训练语言模型的中文命名实体识别，在这里bert类似于word2vec，
去bert模型最后一层bilm的输出结果作为导入到下游任务的词向量。
&ensp;&ensp;下游任务的模型依然是Bi-LSTM + crf。
* 验证集上的结果：f1 = 0.971