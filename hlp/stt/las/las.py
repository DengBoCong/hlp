import tensorflow as tf

from hlp.utils.layers import BahdanauAttention


class Encoder(tf.keras.Model):
    def __init__(self, cnn1_filters, cnn1_kernel_size, cnn2_filters,
                 cnn2_kernel_size, max_pool_strides, max_pool_size, d, w):
        """

        :param cnn1_filters:
        :param cnn1_kernel_size:
        :param cnn2_filters:
        :param cnn2_kernel_size:
        :param max_pool_strides:
        :param max_pool_size:
        :param d: BiLSTM层数
        :param w: BiLSTM单元数
        """
        super(Encoder, self).__init__()
        self.d = d
        self.w = w
        self.cnn1 = tf.keras.layers.Conv1D(filters=cnn1_filters, kernel_size=cnn1_kernel_size, activation='relu')
        self.cnn2 = tf.keras.layers.Conv1D(filters=cnn2_filters, kernel_size=cnn2_kernel_size, activation='relu')
        self.max_pool = tf.keras.layers.MaxPooling1D(strides=max_pool_strides, pool_size=max_pool_size)

        self.bi_lstm = []
        for i in range(self.d):
            self.bi_lstm.append(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(w, return_sequences=True)))

    def call(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.max_pool(x)

        for i in range(self.d):
            x = self.bi_lstm[i](x)

        return x

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.w))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, w):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # TODO: LSTM层数可变
        self.rnn1 = tf.keras.layers.LSTM(w, return_sequences=True)
        # self.rnn2 = tf.keras.layers.LSTM(w, return_sequences=True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        """解码

        :param x: 目标符号, （批大小，id）
        :param hidden: 解码器状态, （批大小，隐藏层大小）
        :param enc_output: 编码器输出, （批大小，最大长度，隐藏层大小）
        :return: token分布, 解码器专题, 注意力权重
        """
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x 在通过嵌入层后的形状 == （批大小，1，嵌入维度）
        x = self.embedding(x)

        # x 在拼接 （concatenation） 后的形状 == （批大小，1，嵌入维度 + 隐藏层大小）
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output = self.rnn1(x)
        # output = self.rnn2(x)
        # 输出的形状 == （批大小 * 1，隐藏层大小）
        output = tf.reshape(output, (-1, output.shape[2]))

        # 输出的形状 == （批大小，vocab）
        tokens_prob = self.fc(output)

        return tokens_prob, attention_weights


class LAS(tf.keras.Model):
    def __init__(self, vocab_tar_size, cnn1_filters, cnn1_kernel_size, cnn2_filters,
                 cnn2_kernel_size, max_pool_strides, max_pool_size, d, w,
                 embedding_dim, dec_units, batch_size):
        super(LAS, self).__init__()
        self.vocab_tar_size = vocab_tar_size
        self.d = d
        self.w = w
        self.batch_size = batch_size
        self.encoder = Encoder(cnn1_filters, cnn1_kernel_size,
                               cnn2_filters, cnn2_kernel_size,
                               max_pool_strides, max_pool_size, d, w)
        self.decoder = Decoder(vocab_tar_size, embedding_dim, dec_units, w)

    def call(self, inputx_1, enc_hidden, dec_input):
        enc_output = self.encoder(inputx_1)

        dec_hidden = enc_hidden  # 编码器状态作为解码器初始状态？
        predictions, dec_hidden = self.decoder(dec_input, dec_hidden, enc_output)
        return predictions, dec_hidden

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.w))
