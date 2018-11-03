import os
import time
import datetime

import numpy as np
import tensorflow as tf

from dataGen import Dataset
from config import config
from charCNN import CharCNN


# 实例化数据生成类
data = Dataset(config.data_source)
# 调用方法生成训练集和测试机
data.dataset_read()

# 生成训练集和验证集
train_x = data.doc_image_train
train_y = data.label_image_train
eval_x = data.doc_image_eval
eval_y = data.label_image_eval


def next_batch(x, y):
    #  输入数据集和标签，生成batch

    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]
    num_batches = (len(x) - 1) // config.batch_size

    for i in range(num_batches - 1):
        start = i * config.batch_size
        end = start + config.batch_size
        batch_x = np.array(x[start: end], dtype="int64")
        batch_y = np.array(y[start: end], dtype="float32")

        yield batch_x, batch_y


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.5  # 配置gpu占用率

    sess = tf.Session(config=session_conf)

    with sess.as_default():
        cnn = CharCNN(sequenceLength=config.sequenceLength,
                      num_classes=config.num_classes,
                      conv_layers=config.model.conv_layers,
                      fc_layers=config.model.fc_layers,
                      data=data
                      )
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # 定义优化函数，传入学习速率参数
        optimizer = tf.train.RMSPropOptimizer(config.model.learning_rate)
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
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Initialize all variables
        saver = tf.train.Saver()
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
            for batch_train in next_batch(train_x, train_y):
                train_step(batch_train[0], batch_train[1])

                current_step = tf.train.global_step(sess, global_step)

                # 对结果进行记录
                if current_step % config.training.evaluate_every == 0:
                    print("\nEvaluation:")
                    losses = []
                    accuracys = []
                    for batch_eval in next_batch(eval_x, eval_y):
                        loss, accuracy = dev_step(batch_eval[0], batch_eval[1])
                        losses.append(loss)
                        accuracys.append(accuracy)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, current_step,
                                                                    sum(losses) / len(losses),
                                                                    sum(accuracys) / len(accuracys)))

                if current_step % config.training.checkpoint_every == 0:
                    path = saver.save(sess, "../model/model_v1/500_middle_hen_augment_1_model/my-model",
                                      global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))