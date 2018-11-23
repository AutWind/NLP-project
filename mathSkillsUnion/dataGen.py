import csv
import json
from collections import Counter
import numpy as np

from config import config
from utils import doc_process, gen_data, open_data


# 处理数据，包括生成训练集和测试集
class Dataset(object):
    def __init__(self, config):
        self.train_data_source = config.train_data_source
        self.eval_data_source = config.eval_data_source
        self.stop_word_source = config.stop_word_source

        self.num_classes = config.num_classes
        self.sequence_length = config.sequence_length
        self.batch_size = config.batch_size
        self.vocab_size = None

        self.stop_word_dict = {}

        self.doc_image_train = []
        self.label_image_train = []
        self.doc_image_eval = []
        self.label_image_eval = []

        self.word_to_index = {}
        self.index_to_word = {}
        self.label_to_index = {}
        self.index_to_label = {}

    def dataset_read(self):
        # 该方法用来初始化生成训练集和测试集

        with open(self.stop_word_source, "r") as fr:
            stop_words = fr.read()
            stop_word_list = stop_words.splitlines()
            self.stop_word_dict = dict(zip(stop_word_list, list(range(len(stop_word_list)))))

        # 打开文件
        train_docs, train_label_name = open_data(self.train_data_source)
        eval_docs, eval_label_name = open_data(self.eval_data_source)

        uniqueLabel = list(set(train_label_name))
        indexLabel = range(len(uniqueLabel))

        self.label_to_index = dict(zip(uniqueLabel, indexLabel))
        self.index_to_label = dict(zip(indexLabel, uniqueLabel))

        with open("../data/data_v2/json/label_to_index.json", "w", encoding="utf-8") as f:
            json.dump(self.label_to_index, f)

        with open("../data/data_v2/json/index_to_label.json", "w", encoding="utf-8") as f:
            json.dump(self.index_to_label, f)

        # 存储用index表示的知识点
        train_label = [self.label_to_index[item] for item in train_label_name]
        eval_label = [self.label_to_index[item] for item in eval_label_name]

        # 词汇表和词的索引表示
        self._gen_vocabulary(train_docs)

        train_doc_image, train_label_image = gen_data(train_docs, train_label, self.word_to_index)
        self.doc_image_train = train_doc_image
        self.label_image_train = train_label_image

        eval_doc_image, eval_label_image = gen_data(eval_docs, eval_label, self.word_to_index)
        self.doc_image_eval = eval_doc_image
        self.label_image_eval = eval_label_image

    def _gen_vocabulary(self, docs):
        # 基于文本信息生成词汇列表
        # 将数据集中所有词读取出来
        all_words = [word for doc in docs for word in doc]
        sub_words = [word for word in all_words if word not in self.stop_word_dict]

        word_count = Counter(sub_words)
        sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

        words = [item[0] for item in sort_word_count if item[1] > 1]
        words = ["pad"] + words + ["UNK"]
        self.vocab_size = len(words)
        self.word_to_index = dict(zip(words, list(range(self.vocab_size))))
        self.index_to_word = dict(zip(list(range(self.vocab_size)), words))

        with open("../data/data_v2/json/word_to_index.json", "w", encoding="utf-8") as f:
            json.dump(self.word_to_index, f)

        with open("../data/data_v2/json/index_to_word.json", "w", encoding="utf-8") as f:
            json.dump(self.index_to_word, f)

    def next_batch(self, x, y):
        #  输入数据集和标签，生成batch

        perm = np.arange(len(x))
        np.random.shuffle(perm)
        x = x[perm]
        y = y[perm]
        num_batches = (len(x) - 1) // self.batch_size

        for i in range(num_batches - 1):
            start = i * config.batch_size
            end = start + config.batch_size
            batch_x = np.array(x[start: end], dtype="int64")
            batch_y = np.array(y[start: end], dtype="float32")

            yield batch_x, batch_y