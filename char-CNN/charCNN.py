from math import sqrt
import tensorflow as tf


# 定义CNN分类器
class CharCNN(object):

    def __init__(self, sequenceLength, num_classes, conv_layers, fc_layers, data):
        # placeholders for input, output and dropuot
        self.input_x = tf.placeholder(tf.int32, [None, sequenceLength], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # 字符嵌入
        with tf.name_scope("embedding"):

            # 获得字符嵌入
            embedding_w = data.onehot_dic_build()[0]
            print("self.w: {}".format(embedding_w.shape))
            self.x_image = tf.nn.embedding_lookup(embedding_w, self.input_x)

            # 添加一个通道维度
            self.x_flat = tf.expand_dims(self.x_image, -1)

        for i, cl in enumerate(conv_layers):
            print("开始第" + str(i + 1) + "卷积层的处理")
            # 利用命名空间name_scope来实现变量名复用
            with tf.name_scope("conv_layer-%s" % (i + 1)):
                filter_width = self.x_flat.get_shape()[2].value
                print(filter_width)
                # filter_shape = [height, width, in_channels, out_channels]
                filter_shape = [cl[1], filter_width, 1, cl[0]]

                stdv = 1 / sqrt(cl[0] * cl[1])

                # 初始化w和b的值
                w_conv = tf.Variable(tf.random_uniform(filter_shape, minval=-stdv, maxval=stdv),
                                     dtype='float32', name='w')
                b_conv = tf.Variable(tf.random_uniform(shape=[cl[0]], minval=-stdv, maxval=stdv), name='b')

                #                 w_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="w")
                #                 b_conv = tf.Variable(tf.constant(0.1, shape=[cl[0]]), name="b")
                # 构建卷积层，可以直接将卷积核的初始化方法传入（w_conv）
                conv = tf.nn.conv2d(self.x_flat, w_conv, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # 加上偏差
                h_conv = tf.nn.bias_add(conv, b_conv)
                # 可以直接加上relu函数，因为tf.nn.conv2d事实上是做了一个卷积运算，然后在这个运算结果上加上偏差，再导入到relu函数中
                # h_conv = tf.nn.relu(h_conv)

                if cl[-1] is not None:
                    ksize_shape = [1, cl[2], 1, 1]
                    h_pool = tf.nn.max_pool(h_conv, ksize=ksize_shape, strides=ksize_shape, padding="VALID",
                                            name="pool")
                else:
                    h_pool = h_conv

                self.x_flat = tf.transpose(h_pool, [0, 1, 3, 2], name="transpose")

        with tf.name_scope("reshape"):
            fc_dim = self.x_flat.get_shape()[1].value * self.x_flat.get_shape()[2].value
            self.x_flat = tf.reshape(self.x_flat, [-1, fc_dim])

        # 保存的是神经元的个数[34*256, 1024, 1024]
        weights = [fc_dim] + fc_layers

        for i, fl in enumerate(fc_layers):
            with tf.name_scope("fc_layer-%s" % (i + 1)):
                print("开始第" + str(i + 1) + "全连接层的处理")
                stdv = 1 / sqrt(weights[i])

                # 定义全连接层的初始化方法，均匀分布初始化w和b的值
                w_fc = tf.Variable(tf.random_uniform([weights[i], fl], minval=-stdv, maxval=stdv), dtype="float32",
                                   name="w")
                b_fc = tf.Variable(tf.random_uniform(shape=[fl], minval=-stdv, maxval=stdv), dtype="float32", name="b")

                #                 w_fc = tf.Variable(tf.truncated_normal([weights[i], fl], stddev=0.05), name="W")
                #                 b_fc = tf.Variable(tf.constant(0.1, shape=[fl]), name="b")

                self.x_flat = tf.nn.relu(tf.matmul(self.x_flat, w_fc) + b_fc)

                with tf.name_scope("drop_out"):
                    self.x_flat = tf.nn.dropout(self.x_flat, self.dropout_keep_prob)

        with tf.name_scope("output_layer"):
            stdv = 1 / sqrt(weights[-1])
            # 定义隐层到输出层的权重系数和偏差的初始化方法
            #             w_out = tf.Variable(tf.truncated_normal([fc_layers[-1], num_classes], stddev=0.1), name="W")
            #             b_out = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            w_out = tf.Variable(tf.random_uniform([fc_layers[-1], num_classes], minval=-stdv, maxval=stdv),
                                dtype="float32", name="w")
            b_out = tf.Variable(tf.random_uniform(shape=[num_classes], minval=-stdv, maxval=stdv), name="b")
            # tf.nn.xw_plus_b就是x和w的乘积加上b
            self.y_pred = tf.nn.xw_plus_b(self.x_flat, w_out, b_out, name="y_pred")
            # tf.argmax函数返回最大值的下标
            self.predictions = tf.argmax(self.y_pred, 1, name="predictions")
            print(self.predictions)

        with tf.name_scope("loss"):
            # 定义损失函数，对预测值进行softmax，再求交叉熵。

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            # tf.equal函数返回整个一个列表，值相同时为True，不同是为False
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")