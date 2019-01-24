import numpy as np
import tensorflow as tf


def predict(char_data, text_data):
    text_data = np.array([text_data])
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            checkpoint_file = tf.train.latest_checkpoint("../model/first_seven_model/")
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            text_x = graph.get_operation_by_name("input_x_text").outputs[0]
            # char_x = graph.get_operation_by_name("input_x_char").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            first_prediction = graph.get_tensor_by_name("first_output/first_predictions:0")
            second_prediction = graph.get_tensor_by_name("second_output/second_predictions:0")
            third_prediction = graph.get_tensor_by_name("third_output/third_predictions:0")

            first_predictions, second_predictions, third_predictions = sess.run(
                [first_prediction, second_prediction, third_prediction],
                feed_dict={text_x: text_data,
                           dropout_keep_prob: 1.0})

            return first_predictions, second_predictions, third_predictions




