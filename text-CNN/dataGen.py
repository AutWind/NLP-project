import csv

import numpy as np

from config import config


# 处理数据，包括生成训练集和测试集
class Dataset(object):
    def __init__(self, config):
        self.train_data_source = config.train_data_source
        self.eval_data_source = config.eval_data_source
        self.test_data_source = config.test_data_source
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
        self.doc_image_test = []
        self.label_image_test = []

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
        train_docs, train_label_name = self._open_data(self.train_data_source)
        eval_docs, eval_label_name = self._open_data(self.eval_data_source)
        test_docs, test_label_name = self._open_data(self.test_data_source)

        uniqueLabel = list(set(train_label_name))
        indexLabel = range(len(uniqueLabel))

        self.label_to_index = dict(zip(uniqueLabel, indexLabel))
        self.index_to_label = dict(zip(indexLabel, uniqueLabel))

        # 存储用index表示的知识点
        train_label = [self.label_to_index[item] for item in train_label_name]
        eval_label = [self.label_to_index[item] for item in eval_label_name]
        test_label = [self.label_to_index[item] for item in test_label_name]

        # 词汇表和词的索引表示
        self._gen_vocabulary(train_docs)

        train_doc_image, train_label_image = self._gen_train_eval_data(train_docs, train_label)
        self.doc_image_train = train_doc_image
        self.label_image_train = train_label_image

        eval_doc_image, eval_label_image = self._gen_train_eval_data(eval_docs, eval_label)
        self.doc_image_eval = eval_doc_image
        self.label_image_eval = eval_label_image

        test_doc_image, test_label_image = self._gen_train_eval_data(test_docs, test_label)
        self.doc_image_test = test_doc_image
        self.label_image_test = test_label_image

    def _open_data(self, filePath):
        """
        读取文件
        :param filePath:
        :return:
        """
        docs = []
        label_name = []
        csvfile = open(filePath, "r")
        for line in csv.reader(csvfile, delimiter=",", quotechar='"'):
            # 把标题和内容组合在一起
            content = line[5].strip().split()
            newContent = [item for item in content if item != ","]
            # docs存储的是所有文本的列表
            docs.append(newContent)
            # label存储的是所有文本对应的类别标签
            label_name.append(line[1])

        docs = docs[1:]
        label_name = label_name[1:]

        return docs, label_name

    def _gen_train_eval_data(self, x, y):

        doc_image = []
        label_image = []
        # 遍历所有的文本，将文本中的词转换成index表示
        for i in range(len(x)):
            doc_vec = self._doc_process(x[i])
            doc_image.append(doc_vec)
            label_class = np.zeros(self.num_classes, dtype="float32")
            label_class[int(y[i]) - 1] = 1
            label_image.append(label_class)

        doc_images = np.asarray(doc_image, dtype="int64")
        label_images = np.array(label_image, dtype="float32")

        return doc_images, label_images

    def _gen_vocabulary(self, docs):
        # 基于文本信息生成词汇列表
        # 将数据集中所有词读取出来
        all_words = [word for doc in docs for word in doc]
        sub_words = [word for word in all_words if word not in self.stop_word_dict]

        word_count = Counter(sub_words)
        sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

        words = [item[0] for item in sort_word_count if item[1] > 3]
        words = ["UNK"] + words
        self.vocab_size = len(words)
        self.word_to_index = dict(zip(words, list(range(self.vocab_size))))
        self.index_to_word = dict(zip(list(range(self.vocab_size)), words))

    def _doc_process(self, doc):
        # 该方法就是讲用词表示的文档转换成用index表示
        doc_vec = np.zeros((self.sequence_length))
        sequence_len = self.sequence_length

        if len(doc) < self.sequence_length:
            sequence_len = len(doc)

        for i in range(sequence_len):
            if doc[i] in self.word_to_index:
                doc_vec[i] = self.word_to_index[doc[i]]
            else:
                doc_vec[i] = self.word_to_index["UNK"]

        return doc_vec

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