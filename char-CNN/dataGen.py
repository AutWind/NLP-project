import csv
import operator

import numpy as np

from config import config


class Dataset(object):
    def __init__(self, data_source):
        self.data_source = data_source
        self.index_in_epoch = 0
        self.alphabet = ""
        self.num_classes = config.num_classes
        self.layer10 = config.sequenceLength
        self.epochs_completed = 0
        self.batch_size = config.batch_size
        self.doc_image_train = []
        self.label_image_train = []
        self.doc_image_eval = []
        self.label_image_eval = []
        self.label_to_index = {}
        self.index_to_label = {}

    def dataset_read(self):
        # 该方法用来初始化生成训练集和测试集
        docs = []
        label_name = []
        doc_count = 0
        csvfile = open(self.data_source, "r")
        for line in csv.reader(csvfile, delimiter=",", quotechar='"'):
            content = line[4]
            # docs存储的是所有文本的列表
            docs.append(content.lower())
            # label存储的是所有文本对应的类别标签
            label_name.append(line[5])
            # doc_count 是统计文本的数量
            doc_count = doc_count + 1

        docs = docs[1:]
        label_name = label_name[1:]
        uniqueLabel = list(set(label_name))
        indexLabel = range(len(uniqueLabel))
        self.label_to_index = dict(zip(uniqueLabel, indexLabel))
        self.index_to_label = dict(zip(indexLabel, uniqueLabel))
        label = [self.label_to_index[item] for item in label_name]

        # 生成字符列表，列表长度为117
        self._gen_alphabet(docs)
        # introduce embedding dict and matrix
        embedding_w, embedding_dic = self.onehot_dic_build()

        doc_image = []
        label_image = []
        # 遍历所有的文本，将文本内容转换成矩阵表示
        for i in range(doc_count - 1):
            doc_vec = self.doc_process(docs[i], embedding_dic)
            doc_image.append(doc_vec)
            label_class = np.zeros(self.num_classes, dtype="float32")
            label_class[int(label[i]) - 1] = 1
            label_image.append(label_class)

        del embedding_w, embedding_dic
        doc_image = np.asarray(doc_image, dtype="int64")
        label_image = np.array(label_image, dtype="float32")

        # 调用内部方法生成训练集和验证集
        self._split_train_test(doc_image, label_image)

    def _split_train_test(self, docs, labels, test_ratio=0.05, shuffle=False):
        # 将原始数据集分割成训练集和测试集，然后保存在实例属性中
        if shuffle:
            shuffled_index = np.random.permutation(range(docs.shape[0]))
            docs = docs[shuffled_index]
            labels = labels[shuffled_index]

        train_size = int(docs.shape[0] * test_ratio)
        self.doc_image_eval = docs[:train_size]
        self.doc_image_train = docs[train_size:]

        self.label_image_eval = labels[:train_size]
        self.label_image_train = labels[train_size:]

    def _gen_alphabet(self, docs):
        # 生成字符列表，然后保存在实例属性中
        longStr = "".join(docs)
        countDict = {}
        for char in longStr:
            if char in countDict.keys():
                countDict[char] += 1
            else:
                countDict[char] = 1
        newCountDict = {}
        for char, value in countDict.items():
            if value > 300:
                newCountDict[char] = value
        newCountDict = sorted(newCountDict.items(), key=operator.itemgetter(1), reverse=True)
        alphabet = [item[0] for item in newCountDict]
        self.alphabet = alphabet[1:]

    def doc_process(self, doc, embedding_dic):
        # 该方法就是讲文本字符串转换成数字表示的字符串，方便之后再字符向量空间中去索引，文本中的空格保留
        # doc是一个文本，embedding_dic是用来寻找字符索引的字典
        min_len = min(self.layer10, len(doc))
        # 设置文本的长度为固定长度1024
        doc_vec = np.zeros(self.layer10, dtype="int64")
        for j in range(min_len):
            # 判断文本中该字符是否在我们的字符字典中
            if doc[j] in embedding_dic:
                # 将doc_vec用一串数字来表示
                doc_vec[j] = embedding_dic[doc[j]]
            else:
                # 对在字符字典中找不到的字符，可以用未知字符来表示
                doc_vec[j] = embedding_dic['UNK']
        return doc_vec

    def onehot_dic_build(self):
        # one-hot encoder

        alphabet = self.alphabet
        embedding_dic = {}
        embedding_w = []
        embedding_dic['UNK'] = 0
        embedding_w.append(np.zeros(len(alphabet), dtype="float32"))
        # 将字符和其索引读取出来
        for i, alpha in enumerate(alphabet):
            onehot = np.zeros(len(alphabet), dtype="float32")
            # 将所有的字符和UNK存储在字典中
            embedding_dic[alpha] = i + 1
            # 生成每个字符对应的onehot向量
            onehot[i] = 1
            # 生成字符嵌入的向量矩阵
            embedding_w.append(onehot)
        embedding_w = np.array(embedding_w, dtype="float32")

        return embedding_w, embedding_dic