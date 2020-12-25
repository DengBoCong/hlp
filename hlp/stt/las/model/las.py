# -*- coding: utf-8 -*-

import tensorflow as tf

from hlp.utils.layers import BahdanauAttention


class Encoder(tf.keras.Model):
    def __init__(self,
                 cnn1_filters,
                 cnn1_kernel_size,
                 cnn2_filters,
                 cnn2_kernel_size,
                 max_pool_strides,
                 max_pool_size,
                 d,
                 w, 
                 batch_sz):
        super(Encoder, self).__init__()
        self.d = d
        self.w = w
        self.batch_sz = batch_sz
        self.cnn1 = tf.keras.layers.Conv1D(filters=cnn1_filters, kernel_size=cnn1_kernel_size, activation='relu')
        self.cnn2 = tf.keras.layers.Conv1D(filters=cnn2_filters, kernel_size=cnn2_kernel_size, activation='relu')
        self.max_pool = tf.keras.layers.MaxPooling1D(strides=max_pool_strides, pool_size=max_pool_size)

        self.bi_lstm = []
        for i in range(self.d):
            self.bi_lstm.append(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(w, return_sequences=True)))

    def call(self, x, hidden):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.max_pool(x)

        for i in range(self.d):
            x = self.bi_lstm[i](x)

        return x, hidden

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.w))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, w):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn1 = tf.keras.layers.LSTM(w, return_sequences=True)
        # self.rnn2 = tf.keras.layers.LSTM(w, return_sequences=True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        # 用于注意力
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # 编码器输出 （enc_output） 的形状 == （批大小，最大长度，隐藏层大小）
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x 在通过嵌入层后的形状 == （批大小，1，嵌入维度）
        x = self.embedding(x)

        # x 在拼接 （concatenation） 后的形状 == （批大小，1，嵌入维度 + 隐藏层大小）
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # 将合并后的向量传送到 RNN
        output = self.rnn1(x)
        # output = self.rnn2(x)

        # 输出的形状 == （批大小 * 1，隐藏层大小）
        output = tf.reshape(output, (-1, output.shape[2]))
        # 输出的形状 == （批大小，vocab）
        x = self.fc(output)

        return x, attention_weights


class LAS(tf.keras.Model):
    def __init__(self,
                 vocab_tar_size,
                 cnn1_filters,
                 cnn1_kernel_size,
                 cnn2_filters,
                 cnn2_kernel_size,
                 max_pool_strides,
                 max_pool_size,
                 d,
                 w,
                 embedding_dim,
                 dec_units,
                 batch_size):
        super(LAS, self).__init__()
        self.vocab_tar_size = vocab_tar_size
        self.d = d
        self.w = w
        self.batch_size = batch_size
        self.encoder = Encoder(cnn1_filters,
                               cnn1_kernel_size,
                               cnn2_filters,
                               cnn2_kernel_size,
                               max_pool_strides,
                               max_pool_size,
                               d,
                               w,
                               batch_size)
        self.decoder = Decoder(vocab_tar_size, embedding_dim, dec_units, w)

    def call(self, inputx_1, enc_hidden, dec_input, candi_size=1):
        enc_output, enc_hidden = self.encoder(inputx_1, enc_hidden)  # 前向计算，编码

        dec_hidden = enc_hidden  # 编码器状态作为解码器初始状态？
        # 根据当前候选结果数复制相同数量的enc_output和dec_hidden，喂入decoder中
        enc_outputs = enc_output
        dec_hiddens = dec_hidden
        for i in range(candi_size - 1):
            enc_outputs = tf.concat([enc_outputs, enc_output], 0)
            dec_hiddens = tf.concat([dec_hiddens, dec_hidden], 0)
        predictions, _ = self.decoder(dec_input, dec_hiddens, enc_outputs)
        return predictions, dec_hidden

    def initialize_hidden_state(self):
        return self.encoder.initialize_hidden_state()