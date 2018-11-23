import tensorflow as tf
from tensorflow.python.layers.core import Dense


# 定义模型
class Seq2SeqModel(object):

    def __init__(self, config, encoder_vocab_size, target_char_to_int, is_infer=False):
        self.inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
        self.targets = tf.placeholder(tf.int32, [None, None], name="targets")
        self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")
        self.source_sequence_length = tf.placeholder(tf.int32, [None], name="source_sequence_length")
        self.target_sequence_length = tf.placeholder(tf.int32, [None], name="target_sequence_length")
        self.target_max_length = tf.reduce_max(self.target_sequence_length, name='target_max_length')

        decoder_output = self.seq2seq(config, encoder_vocab_size, target_char_to_int, is_infer)

        if is_infer:
            self.infer_logits = tf.identity(decoder_output.sample_id, "infer_logits")

        else:
            self.logits = tf.identity(decoder_output.rnn_output, "logits")

            masks = tf.sequence_mask(self.target_sequence_length, self.target_max_length, dtype=tf.float32, name="mask")

            with tf.name_scope("loss"):
                self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.targets, masks)

            with tf.name_scope("accuracy"):
                self.predictions = tf.argmax(self.logits, 2)
                correctness = tf.equal(tf.cast(self.predictions, dtype=tf.int32), self.targets)
                self.accu = tf.reduce_mean(tf.cast(correctness, "float"), name="accu")

    def encoder(self, config, encoder_vocab_size):
        encoder_embed_input = tf.contrib.layers.embed_sequence(self.inputs, encoder_vocab_size,
                                                               config.model.encoder_embedding_size)

        def get_lstm_cell(hidden_size):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True,
                                                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.dropout_prob)

            return drop_cell

        cell = tf.nn.rnn_cell.MultiRNNCell(
            [get_lstm_cell(hidden_size) for hidden_size in config.model.encoder_hidden_layers])
        outputs, final_state = tf.nn.dynamic_rnn(cell, encoder_embed_input, sequence_length=self.source_sequence_length,
                                                 dtype=tf.float32)

        return outputs, final_state

    def decoder(self, config, encoder_state, target_char_to_int, is_infer):

        decoder_vocab_size = len(target_char_to_int)

        embeddings = tf.Variable(tf.random_uniform([decoder_vocab_size, config.model.decoder_embedding_size]))
        decoder_embed_input = tf.nn.embedding_lookup(embeddings, self.targets)

        def get_lstm_cell(hidden_size):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True,
                                                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.dropout_prob)

            return drop_cell

        cell = tf.nn.rnn_cell.MultiRNNCell(
            [get_lstm_cell(hidden_size) for hidden_size in config.model.decoder_hidden_layers])

        # 定义有Dense方法生成的全连接层
        output_layer = Dense(decoder_vocab_size,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        # 定义训练时的decode的代码
        with tf.variable_scope("decode"):
            # 得到help对象，帮助读取数据
            train_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                             sequence_length=self.target_sequence_length)

            # 构建decoder
            train_decoder = tf.contrib.seq2seq.BasicDecoder(cell, train_helper, encoder_state, output_layer)
            train_decoder_output, train_state, train_sequence_length = tf.contrib.seq2seq.dynamic_decode(
                train_decoder, impute_finished=True, maximum_iterations=self.target_max_length)

        # 定义预测时的decode代码
        with tf.variable_scope("decode", reuse=True):
            # 解码时的第一个时间步上的输入，之后的时间步上的输入是上一时间步的输出
            start_tokens = tf.tile(tf.constant([target_char_to_int["<GO>"]], dtype=tf.int32), [config.batch_size],
                                   name="start_tokens")

            # 解码时按贪心法解码，按照最大条件概率来预测输出值，该方法需要输入启动词和结束词，启动词是个一维tensor，结束词是标量
            infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens,
                                                                    target_char_to_int["<EOS>"])
            infer_decoder = tf.contrib.seq2seq.BasicDecoder(cell, infer_helper, encoder_state, output_layer)
            infer_decoder_output, infer_state, infer_sequence_length = tf.contrib.seq2seq.dynamic_decode(
                infer_decoder, impute_finished=True, maximum_iterations=self.target_max_length)

        if is_infer:
            return infer_decoder_output

        return train_decoder_output

    def seq2seq(self, config, encoder_vocab_size, target_char_to_int, is_infer):
        """
        将encoder和decoder合并输出
        """
        encoder_output, encoder_state = self.encoder(config, encoder_vocab_size)

        decoder_output = self.decoder(config, encoder_state, target_char_to_int, is_infer)

        return decoder_output