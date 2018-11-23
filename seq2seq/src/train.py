import tensorflow as tf

from utils import next_batch
from config import Config
from dataGen import DataGen
from seq2seqModel import Seq2SeqModel


# 训练模型

class Engine():
    def __init__(self):
        self.config = Config()
        self.dataGen = DataGen(self.config)
        self.dataGen.read_data()
        self.sess = None
        self.global_step = 0

    def train_step(self, sess, train_op, train_model, params):

        feed_dict = {
            train_model.inputs: params["source_batch"],
            train_model.targets: params["target_batch"],
            train_model.dropout_prob: self.config.model.dropout_prob,
            train_model.source_sequence_length: params["source_sequence_length"],
            train_model.target_sequence_length: params["target_sequence_length"],
        }

        _, loss, accu = sess.run([train_op, train_model.loss, train_model.accu], feed_dict)

        return loss, accu

    def infer_step(self, sess, infer_model, params):

        feed_dict = {
            infer_model.inputs: params["source_batch"],
            infer_model.targets: params["target_batch"],
            infer_model.dropout_prob: 1.0,
            infer_model.source_sequence_length: params["source_sequence_length"],
            infer_model.target_sequence_length: params["target_sequence_length"],
        }

        logits = sess.run([infer_model.infer_logits], feed_dict)
        predictions = logits[0]

        prediction = [sequence[:end] for sequence in predictions for end in params["target_sequence_length"]]
        target = [sequence[:end] for sequence in params["target_batch"] for end in params["target_sequence_length"]]

        total = 0
        correct = 0
        for i in range(len(prediction)):
            for j in range(len(prediction[i])):
                if prediction[i][j] == target[i][j]:
                    correct += 1
            total += len(prediction[i])

        accu = correct / total

        return accu

    def run_epoch(self):
        config = self.config
        dataGen = self.dataGen

        source_data = dataGen.source_data
        target_data = dataGen.target_data

        train_split = int(len(source_data) * config.infer_prob)

        train_source_data = source_data[train_split:]
        infer_source_data = source_data[: train_split]

        train_target_data = target_data[train_split:]
        infer_target_data = target_data[: train_split]

        source_char_to_int = dataGen.source_char_to_int
        target_char_to_int = dataGen.target_char_to_int

        encoder_vocab_size = len(source_char_to_int)

        batch_size = config.batch_size

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                with tf.name_scope("train"):
                    with tf.variable_scope("seq2seq"):
                        train_model = Seq2SeqModel(config, encoder_vocab_size, target_char_to_int, is_infer=False)

                with tf.name_scope("infer"):
                    with tf.variable_scope("seq2seq", reuse=True):
                        infer_model = Seq2SeqModel(config, encoder_vocab_size, target_char_to_int, is_infer=True)

                global_step = tf.Variable(0, name="global_step", trainable=False)

                optimizer = tf.train.AdamOptimizer(config.train.learning_rate)
                grads_and_vars = optimizer.compute_gradients(train_model.loss)
                grads_and_vars = [(tf.clip_by_norm(g, config.train.max_grad_norm), v) for g, v in grads_and_vars if
                                  g is not None]
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step, name="train_op")

                saver = tf.train.Saver(tf.global_variables())
                sess.run(tf.global_variables_initializer())

                print("初始化完成，开始训练模型")
                for i in range(config.train.epochs):
                    for params in next_batch(train_source_data, train_target_data, batch_size, source_char_to_int,
                                             target_char_to_int):
                        loss, accu = self.train_step(sess, train_op, train_model, params)
                        current_step = tf.train.global_step(sess, global_step)
                        print("step: {}  loss: {}  accu: {}".format(current_step, loss, accu))

                        if current_step % config.train.every_checkpoint == 0:
                            accus = []
                            for params in next_batch(infer_source_data, infer_target_data, batch_size,
                                                     source_char_to_int, target_char_to_int):
                                accu = self.infer_step(sess, infer_model, params)
                                accus.append(accu)
                            print("\n")
                            print("Evaluation accuracy: {}".format(sum(accus) / len(accus)))
                            print("\n")
                            saver.save(sess, "model/my-model", global_step=current_step)


if __name__ == "__main__":
    engine = Engine()
    engine.run_epoch()