import csv
import numpy as np
from config import config


def open_data(filePath):
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


def doc_process(doc, sequence_length, word_to_index):
    # 该方法就是讲用词表示的文档转换成用index表示
    doc_vec = np.zeros((sequence_length))
    sequence_len = sequence_length

    if len(doc) < sequence_length:
        sequence_len = len(doc)

    for i in range(sequence_len):
        if doc[i] in word_to_index:
            doc_vec[i] = word_to_index[doc[i]]
        else:
            doc_vec[i] = word_to_index["UNK"]

    return doc_vec


def gen_data(x, y, word_to_index):

    doc_image = []
    label_image = []
    # 遍历所有的文本，将文本中的词转换成index表示
    for i in range(len(x)):
        doc_vec = doc_process(x[i], config.sequence_length, word_to_index)
        doc_image.append(doc_vec)
        label_class = np.zeros(config.num_classes, dtype="float32")
        label_class[int(y[i])] = 1
        label_image.append(label_class)

    doc_images = np.asarray(doc_image, dtype="int64")
    label_images = np.array(label_image, dtype="float32")

    return doc_images, label_images