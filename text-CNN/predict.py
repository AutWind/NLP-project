import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from dataGen import Dataset
from config import config
from train import next_batch


# 实例化数据生成类
data = Dataset(config.data_source)
# 调用方法生成训练集和测试机
data.dataset_read()


def predict():
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            checkpoint_file = tf.train.latest_checkpoint("../model/model_v2/seven_model/")
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            prediction = graph.get_operation_by_name("output/predictions").outputs[0]

            predictions = []

            predictions.extend(sess.run(prediction, feed_dict={x: data.doc_image_test, dropout_keep_prob: 1.0}))

            for batch_test in data.next_batch(data.doc_image_eval, data.label_image_eval):
                prediction_ = sess.run(prediction, feed_dict={x: batch_test[0], dropout_keep_prob:1.0})
                predictions = np.concatenate([predictions, prediction_])

            y_true = np.argmax(data.label_image_test, axis=1)
            labels = list(set(y_true))

            accu = np.mean(np.equal(predictions, y_true))

            confusionMat = confusion_matrix(y_true, predictions, labels)

            return y_true, predictions, accu, confusionMat, labels


y_true, predictions, accu, confusionMat, labels = predict()