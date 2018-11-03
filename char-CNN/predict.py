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
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("../model/model_v1/500_middle_hen_augment_1_model/my-model-12600.meta")
        saver.restore(sess, tf.train.latest_checkpoint("../model/model_v1/500_middle_hen_augment_1_model/"))
        graph = tf.get_default_graph()
        x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        prediction = graph.get_tensor_by_name("output_layer/predictions:0")

        predictions = []

        for batch_test in next_batch(data.doc_image_eval, data.label_image_eval):
            prediction_ = sess.run(prediction, feed_dict={x: batch_test[0], dropout_keep_prob:1.0})
            predictions.extend(prediction_)

        y_true = np.argmax(data.label_image_eval, axis=1)[:len(predictions)]
        labels = list(set(y_true))
        accu = np.mean(np.equal(predictions, y_true))

        confusionMat = confusion_matrix(y_true, predictions, labels)

        return accu, confusionMat


accu, confusionMat = predict()