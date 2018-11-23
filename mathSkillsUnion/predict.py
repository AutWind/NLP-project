import json
import numpy as np
import tensorflow as tf

from utils import open_data, gen_data

with open('../data/data_v2/json/label_to_index.json', encoding='utf-8') as f:
    label_to_index = json.load(f)

with open('../data/data_v2/json/index_to_label.json', encoding='utf-8') as f:
    index_to_label = json.load(f)

with open('../data/data_v2/json/word_to_index.json', encoding='utf-8') as f:
    word_to_index = json.load(f)


def gen_test_data():
    docs, label_name = open_data("../data/data_v2/500_token_seven_test.csv")

    labels = [label_to_index[item] for item in label_name]

    doc_test, label_test = gen_data(docs, labels, word_to_index)

    return doc_test, label_test


def predict():
    doc_test, label_test = gen_test_data()

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            checkpoint_file = tf.train.latest_checkpoint("../model/model_v2/500_middle_model/")
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            prediction = graph.get_tensor_by_name("output/predictions:0")

            test_predictions = sess.run(prediction, feed_dict={x: doc_test, dropout_keep_prob: 1.0})
            test_y_true = np.argmax(label_test, axis=1)
            test_accu = np.mean(np.equal(test_predictions, test_y_true))

            return test_y_true, test_predictions, test_accu


test_y_true, test_predictions, test_accu = predict()