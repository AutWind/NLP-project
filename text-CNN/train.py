import os
import time
import datetime

import numpy as np
import tensorflow as tf

from dataGen import Dataset
from config import config
from textCNN import TextCNN


# 实例化数据生成类
data = Dataset(config)
# 调用方法生成训练集和测试机
data.dataset_read()

# 生成训练集和验证集
raw_train_x = data.doc_image_train
raw_train_y = data.label_image_train
# print(raw_train_x.shape)
# print(raw_train_y.shape)
# aug_train_x, aug_train_y = augment_data(raw_train_x, raw_train_y)

# print(aug_train_x.shape)
# print(aug_train_y.shape)
# train_x = np.vstack((raw_train_x, aug_train_x))
# train_y = np.vstack((raw_train_y, aug_train_y))

# print(train_x.shape)
# print(train_y.shape)

train_x = raw_train_x
train_y = raw_train_y

eval_x = data.doc_image_eval
eval_y = data.label_image_eval

vocab_size = data.vocab_size

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.5  # 配置gpu占用率

    sess = tf.Session(config=session_conf)

    with sess.as_default():
        cnn = TextCNN(config, vocab_size)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # 定义优化函数，传入学习速率参数
        optimizer = tf.train.AdamOptimizer(config.training.learning_rate)
        # 计算梯度,得到梯度和变量
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        # 将梯度应用到变量下，生成训练器
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summariescd
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Initialize all variables
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: config.model.dropout_keep_prob
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            train_summary_writer.add_summary(train_summary_op, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, loss, accuracy = sess.run(
                [global_step, cnn.loss, cnn.accuracy],
                feed_dict)

            return loss, accuracy

        for i in range(config.training.epoches):
            # 训练模型
            print("start training model")
            for batch_train in data.next_batch(train_x, train_y):
                train_step(batch_train[0], batch_train[1])

                current_step = tf.train.global_step(sess, global_step)
                if current_step % config.training.evaluate_every == 0:
                    print("\nEvaluation:")
                    losses = []
                    accuracys = []
                    for batch_eval in data.next_batch(eval_x, eval_y):
                        loss, accuracy = dev_step(batch_eval[0], batch_eval[1])
                        losses.append(loss)
                        accuracys.append(accuracy)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, current_step,
                                                                    sum(losses) / len(losses),
                                                                    sum(accuracys) / len(accuracys)))

                if current_step % config.training.checkpoint_every == 0:
                    path = saver.save(sess, "../model/model_v2/seven_model/my-model", global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))