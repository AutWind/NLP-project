import re
import sys
import jieba
import json
from bs4 import BeautifulSoup
from flask import Flask
from flask import request
from pypinyin import pinyin
import pypinyin
sys.path.append("../src/")

from dataGen import doc_process
from config import config
from prediction_show import predict

app = Flask(__name__)

with open('../data/first_json/third_label_to_index.json', encoding='utf-8') as f:
    third_label_to_index = json.load(f)

with open('../data/first_json/word_to_index.json', encoding='utf-8') as f:
    word_to_index = json.load(f)

with open('../data/first_json/embedding_dic.json', encoding='utf-8') as f:
    embedding_dic = json.load(f)

third_index_to_label = dict(zip(list(third_label_to_index.values()), list(third_label_to_index.keys())))
first_index_to_label = dict(zip(list(config.first_label_to_index.values()), list(config.first_label_to_index.keys())))
second_index_to_label = dict(zip(list(config.second_label_to_index.values()), list(config.second_label_to_index.keys())))

labels = list(third_label_to_index.keys())
for label in labels:
    jieba.suggest_freq(label, True)

"""
http://127.0.0.1:5000/getQuestionTags?content=xx&analyse=yy&discuss=cc
"""


# @app.route("/getQuestionTags", methods=["POST", "GET"])
def getQuestionTags():
    # content = request.args.get("content", "")
    # analyse = request.args.get("analyse", "")
    # discuss = request.args.get("discuss", "")
    content = "<img alt=\"菁优网\" src=\"/2e51dbcfeeba41429c224c6aafc28378.png\" data-iobs-key=\"2e51dbcfeeba41429c224c6aafc28378.png\" style=\"vertical-align:middle;FLOAT:right\" />如图△ABC中，DE是BC的垂直平分线，△ABD的周长为7cm，BE=2cm，则△ABC的周长为<!--BA--><div class='quizPutTag' contenteditable='true'>&nbsp;</div><!--EA-->cm"
    analyse = "根据线段的垂直平分线的性质得到DB=DC，CE=BC=2，根据三角形的周长公式计算即可"
    discuss = "本题考查的是线段的垂直平分线的性质，掌握线段的垂直平分线上的点到线段的两个端点的距离相等是解题的关键"

    charData, textData = tackleData(content, analyse, discuss)
    char_data = doc_process(charData, config.charCNN.sequence_length, embedding_dic)
    text_data = doc_process(textData, config.textCNN.sequence_length, word_to_index)
    first_index, second_index, third_index = predict(char_data, text_data)

    first_label = first_index_to_label[first_index[0]]
    second_label = second_index_to_label[second_index[0]]
    third_label = third_index_to_label[third_index[0]]

    return first_label, second_label, third_label


def tackleData(content, analyse, discuss):
    charContent = char_tackle(content)
    tokenContent = text_tackle(content)
    tokenAnalyse = text_tackle(analyse)
    tokenDiscuss = text_tackle(discuss)
    tokenDocument = tokenContent + tokenAnalyse + tokenDiscuss

    return charContent, tokenDocument


def char_tackle(subject):
    beau = BeautifulSoup(subject)
    newSubject = beau.get_text()
    newSubject = re.sub("（\u3000\u3000）", "", newSubject)
    newSubject = re.sub("\xa0", "", newSubject)
    newSubject = re.sub("\u3000", "", newSubject)
    newSubject = re.sub("&bnsp;", "", newSubject)
    newSubject = pinyin(newSubject, style=pypinyin.TONE2)
    newSubject = [item[0] for item in newSubject]
    newSubject = " ".join(newSubject)

    return newSubject


def text_tackle(subject):
    p2 = re.compile(r'[^\u4e00-\u9fa5]')
    newSubject = " ".join(p2.split(subject)).strip()
    newSubject = ",".join(newSubject.split())
    newSubject = re.sub("菁优网,", "", newSubject)
    newSubject = tokenizer(newSubject)

    return newSubject


def tokenizer(subject):
    newSubject = list(jieba.cut(subject))
    newSubject = [item for item in newSubject if item != ","]

    return newSubject


first_label, second_label, third_label = getQuestionTags()
print(first_label)
print(second_label)
print(third_label)