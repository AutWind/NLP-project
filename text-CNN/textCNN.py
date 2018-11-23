import tensorflow as tf


# 构建模型
class EnsembleCNN(object):
    """
    textCNN 和 charCNN的集成融合，配合目标输出的多任务学习
    """

    def __init__(self, config, vocab_size):

        # Placeholders for input, output and dropout
        self.input_x_text = tf.placeholder(tf.int32, [None, config.textCNN.sequence_length], name="input_x_text")
        self.input_x_char = tf.placeholder(tf.int32, [None, config.charCNN.sequence_length], name="input_x_char")

        self.input_y_first = tf.placeholder(tf.float32, [None, config.first_num_classes], name="input_y_first")
        self.input_y_second = tf.placeholder(tf.float32, [None, config.second_num_classes], name="input_y_second")
        self.input_y_third = tf.placeholder(tf.float32, [None, config.third_num_classes], name="input_y_third")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.vocab_size = vocab_size
        self.config = config

        # Keeping track of l2 regularization loss (optional)
        first_l2_loss = tf.constant(0.0)
        second_l2_loss = tf.constant(0.0)
        third_l2_loss = tf.constant(0.0)

        text_pool_flat = self.textCNN()
        char_pool_flat = self.charCNN()

        concat_pool_flat = tf.concat([text_pool_flat, char_pool_flat], 1)
        concat_pool_flat = text_pool_flat
        pool_flat_size = concat_pool_flat.get_shape()[1].value

        # Final (unnormalized) scores and predictions
        with tf.name_scope("first_output"):
            first_W = tf.get_variable(
                "first_W",
                shape=[pool_flat_size, config.first_num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            first_b = tf.Variable(tf.constant(0.1, shape=[config.first_num_classes]), name="first_b")
            first_l2_loss += tf.nn.l2_loss(first_W)
            first_l2_loss += tf.nn.l2_loss(first_b)
            self.first_scores = tf.nn.xw_plus_b(concat_pool_flat, first_W, first_b, name="first_scores")
            self.first_predictions = tf.argmax(self.first_scores, 1, name="first_predictions")

        with tf.name_scope("second_output"):
            second_W = tf.get_variable(
                "second_W",
                shape=[pool_flat_size, config.second_num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            second_b = tf.Variable(tf.constant(0.1, shape=[config.second_num_classes]), name="second_b")
            second_l2_loss += tf.nn.l2_loss(second_W)
            second_l2_loss += tf.nn.l2_loss(second_b)
            self.second_scores = tf.nn.xw_plus_b(concat_pool_flat, second_W, second_b, name="second_scores")
            self.second_predictions = tf.argmax(self.second_scores, 1, name="second_predictions")

        with tf.name_scope("third_output"):
            third_W = tf.get_variable(
                "third_W",
                shape=[pool_flat_size, config.third_num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            third_b = tf.Variable(tf.constant(0.1, shape=[config.third_num_classes]), name="third_b")
            third_l2_loss += tf.nn.l2_loss(third_W)
            third_l2_loss += tf.nn.l2_loss(third_b)
            self.third_scores = tf.nn.xw_plus_b(concat_pool_flat, third_W, third_b, name="third_scores")
            self.third_predictions = tf.argmax(self.third_scores, 1, name="third_predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            first_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.first_scores, labels=self.input_y_first)
            second_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.second_scores,
                                                                    labels=self.input_y_second)
            third_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.third_scores, labels=self.input_y_third)
            losses_concat = tf.concat([first_losses, second_losses, third_losses], 0)

            self.loss = tf.reduce_mean(
                losses_concat) + config.second_l2_lambda * second_l2_loss + config.first_l2_lambda * first_l2_loss

        # Accuracy
        with tf.name_scope("first_accuracy"):
            correct_predictions = tf.equal(self.first_predictions, tf.argmax(self.input_y_first, 1))
            self.first_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="first_accuracy")

        with tf.name_scope("second_accuracy"):
            correct_predictions = tf.equal(self.second_predictions, tf.argmax(self.input_y_second, 1))
            self.second_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="second_accuracy")

        with tf.name_scope("third_accuracy"):
            correct_predictions = tf.equal(self.third_predictions, tf.argmax(self.input_y_third, 1))
            self.third_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="third_accuracy")

    def charCNN(self):

        with tf.name_scope("char_embedding"):
            embedding_w = char_data.onehot_dic_build()[0]
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            embedded_char = tf.nn.embedding_lookup(embedding_w, self.input_x_char)
            # 将输入数据摊平，之后输入到tf.nn.conv2d中
            embedding_char_expanded = tf.expand_dims(embedded_char, -1)

        for i, cl in enumerate(config.charCNN.conv_layers):
            print("开始第" + str(i + 1) + "卷积层的处理")
            # 利用命名空间name_scope来实现变量名复用
            with tf.name_scope("char_conv_layer-%s" % (i + 1)):
                filter_width = embedding_char_expanded.get_shape()[2].value

                # filter_shape = [height, width, in_channels, out_channels]
                filter_shape_char = [cl[1], filter_width, 1, cl[0]]

                stdv = 1 / sqrt(cl[0] * cl[1])

                # 初始化w和b的值
                w_conv = tf.Variable(tf.random_uniform(filter_shape_char, minval=-stdv, maxval=stdv),
                                     dtype='float32', name='char_w')
                b_conv = tf.Variable(tf.random_uniform(shape=[cl[0]], minval=-stdv, maxval=stdv), name='char_b')

                #                 w_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="w")
                #                 b_conv = tf.Variable(tf.constant(0.1, shape=[cl[0]]), name="b")
                # 构建卷积层，可以直接将卷积核的初始化方法传入（w_conv）
                conv = tf.nn.conv2d(embedding_char_expanded, w_conv, strides=[1, 1, 1, 1], padding="VALID", name="conv")
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

                embedding_char_expanded = tf.transpose(h_pool, [0, 1, 3, 2], name="transpose")

        with tf.name_scope("char_reshape"):
            fc_dim = embedding_char_expanded.get_shape()[1].value * embedding_char_expanded.get_shape()[2].value
            pool_flat = tf.reshape(embedding_char_expanded, [-1, fc_dim])

        # 保存的是神经元的个数[34*256, 1024, 1024]
        weights = [fc_dim] + config.charCNN.fc_layers

        for i, fl in enumerate(config.charCNN.fc_layers):
            with tf.name_scope("char_fc_layer-%s" % (i + 1)):
                print("开始第" + str(i + 1) + "全连接层的处理")
                stdv = 1 / sqrt(weights[i])

                # 定义全连接层的初始化方法，均匀分布初始化w和b的值
                w_fc = tf.Variable(tf.random_uniform([weights[i], fl], minval=-stdv, maxval=stdv), dtype="float32",
                                   name="w")
                b_fc = tf.Variable(tf.random_uniform(shape=[fl], minval=-stdv, maxval=stdv), dtype="float32", name="b")

                #                 w_fc = tf.Variable(tf.truncated_normal([weights[i], fl], stddev=0.05), name="W")
                #                 b_fc = tf.Variable(tf.constant(0.1, shape=[fl]), name="b")

                pool_flat = tf.nn.relu(tf.matmul(pool_flat, w_fc) + b_fc)

                with tf.name_scope("drop_out"):
                    pool_flat = tf.nn.dropout(pool_flat, self.dropout_keep_prob)

        return pool_flat

    def textCNN(self):
        with tf.name_scope("text_embedding"):
            # 利用均匀分布初始化词嵌入矩阵
            self.embedding_w = tf.Variable(
                tf.random_uniform([self.vocab_size, self.config.textCNN.embedding_size], -1.0, 1.0),
                name="text_w")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            embedded_chars = tf.nn.embedding_lookup(self.embedding_w, self.input_x_text)
            # 将输入数据摊平，之后输入到tf.nn.conv2d中
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        # 创建卷积和池化层
        pooled_outputs = []
        for i, filter_size in enumerate(self.config.textCNN.filter_sizes):
            with tf.name_scope("text_conv-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.config.textCNN.embedding_size, 1, self.config.textCNN.num_filters]
                conv_w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_w")
                conv_b = tf.Variable(tf.constant(0.1, shape=[self.config.textCNN.num_filters]), name="conv_b")
                conv = tf.nn.conv2d(
                    embedded_chars_expanded,
                    conv_w,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                conv_h = tf.nn.relu(tf.nn.bias_add(conv, conv_b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    conv_h,
                    ksize=[1, self.config.textCNN.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features

        h_pool = tf.concat(pooled_outputs, 3)
        pool_flat = tf.reshape(h_pool, [-1, h_pool.get_shape()[3].value])

        return pool_flat